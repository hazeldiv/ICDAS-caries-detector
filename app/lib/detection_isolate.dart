// detection_isolate.dart
//
// All heavy-lifting runs here in a long-lived background isolate so the
// main (UI) thread — and the camera preview — are never blocked.
//
// Data flow:
//   Main → (FrameRequest via SendPort) → Isolate
//   Isolate → (DetectionResult via SendPort) → Main
//
// The combined single-pass YUV→letterbox→Float32 loop does:
//   1. Compute the sensor pixel for each 640×640 model pixel using the
//      inverse letterbox + inverse 90° CW rotation math.
//   2. Read the YUV values directly from the sensor planes.
//   3. Write the normalised RGB floats into the pre-allocated input buffer.
// This avoids intermediate image copies and is as fast as possible in Dart.

import 'dart:isolate';
import 'dart:typed_data';
import 'dart:math' as math;

import 'package:tflite_flutter/tflite_flutter.dart';

// ════════════════════════════════════════════════════════════════════════════
// Public data classes
// All fields are Dart primitives / typed lists → safely sendable between isolates.
// ════════════════════════════════════════════════════════════════════════════

/// Letterbox geometry used both to build the model input and to unmap boxes.
class LetterboxParams {
  final double scale;
  final double padX;  // horizontal padding in model pixels (left & right)
  final double padY;  // vertical padding in model pixels (top & bottom)
  final int origW;    // portrait display width  (sensor height after CW rotation)
  final int origH;    // portrait display height (sensor width  after CW rotation)

  const LetterboxParams({
    required this.scale,
    required this.padX,
    required this.padY,
    required this.origW,
    required this.origH,
  });
}

/// A single detected bounding box in portrait-display-normalised coordinates [0..1].
/// Uses plain doubles instead of dart:ui Rect to guarantee isolate sendability.
class Detection {
  final double left;
  final double top;
  final double right;
  final double bottom;
  final double confidence;
  final int classId;
  final String label;

  const Detection({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
    required this.confidence,
    required this.classId,
    required this.label,
  });
}

/// Raw camera frame data extracted from a CameraImage on the main isolate.
/// Every field is a primitive or Uint8List — safe to send across isolates.
class FrameRequest {
  // YUV planes (Android 3-plane) or Y+UV (iOS 2-plane) or BGRA (1-plane).
  final Uint8List yBytes;
  final Uint8List uBytes;
  final Uint8List vBytes;   // empty List for iOS 2-plane; same as uBytes ref is fine

  final int imageWidth;     // sensor landscape width  (e.g. 1280)
  final int imageHeight;    // sensor landscape height (e.g.  720)

  // Stride/pixel-step metadata for each plane.
  final int yRowStride;
  final int yPixelStride;
  final int uvRowStride;
  final int uvPixelStride;
  final int vRowStride;
  final int vPixelStride;

  /// 1 = BGRA, 2 = iOS semi-planar YUV, 3 = Android YUV420
  final int planeCount;

  // Per-frame inference settings (so slider changes take effect immediately).
  final bool isFrontCamera;
  final int sensorOrientation;
  final double confidenceThreshold;
  final bool targetOnly;

  const FrameRequest({
    required this.yBytes,
    required this.uBytes,
    required this.vBytes,
    required this.imageWidth,
    required this.imageHeight,
    required this.yRowStride,
    required this.yPixelStride,
    required this.uvRowStride,
    required this.uvPixelStride,
    required this.vRowStride,
    required this.vPixelStride,
    required this.planeCount,
    required this.isFrontCamera,
    required this.sensorOrientation,
    required this.confidenceThreshold,
    required this.targetOnly,
  });
}

/// Results posted back from the detection isolate to the main thread.
class DetectionResult {
  final List<Detection> detections;
  final int inferenceMs;      // total wall-clock time inside the isolate (ms)
  final LetterboxParams lbp;  // for display/debug info

  const DetectionResult({
    required this.detections,
    required this.inferenceMs,
    required this.lbp,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// Isolate bootstrap
// ════════════════════════════════════════════════════════════════════════════

/// Sent once to the isolate when it is spawned.
class IsolateInitPayload {
  final Uint8List modelBytes;
  final SendPort mainSendPort;
  IsolateInitPayload({required this.modelBytes, required this.mainSendPort});
}

/// Top-level entry point for Isolate.spawn().
/// Creates its own Interpreter from the raw model bytes, then listens
/// forever for FrameRequest messages.
void detectionIsolateEntry(IsolateInitPayload payload) {
  // Build interpreter inside the isolate (cannot cross isolate boundaries).
  final interpreter = Interpreter.fromBuffer(
    payload.modelBytes,
    options: InterpreterOptions()..threads = 4,
  );
  interpreter.allocateTensors();

  // Pre-allocated 640×640×3 Float32 input buffer — reused every frame to
  // eliminate per-frame heap allocation pressure.
  final inputBuffer = Float32List(640 * 640 * 3);

  // Create our receive port and hand its SendPort to the main isolate
  // as the first (handshake) message.
  final receivePort = ReceivePort();
  payload.mainSendPort.send(receivePort.sendPort);

  receivePort.listen((message) {
    if (message is FrameRequest) {
      DetectionResult result;
      try {
        result = _processFrame(message, interpreter, inputBuffer);
      } catch (e) {
        // Always post back so _isolateBusy is reset in main.
        result = DetectionResult(
          detections: const [],
          inferenceMs: 0,
          lbp: LetterboxParams(
            scale: 1, padX: 0, padY: 0,
            origW: 640, origH: 640,
          ),
        );
      }
      payload.mainSendPort.send(result);
    } else if (message == 'shutdown') {
      interpreter.close();
      receivePort.close();
    }
  });
}

// ════════════════════════════════════════════════════════════════════════════
// Core frame processing (runs entirely inside the background isolate)
// ════════════════════════════════════════════════════════════════════════════

DetectionResult _processFrame(
  FrameRequest req,
  Interpreter interpreter,
  Float32List inputBuffer,
) {
  final stopwatch = Stopwatch()..start();

  // Compute letterbox geometry based on sensor orientation.
  final int portW = (req.sensorOrientation == 90 || req.sensorOrientation == 270)
      ? req.imageHeight
      : req.imageWidth;
  final int portH = (req.sensorOrientation == 90 || req.sensorOrientation == 270)
      ? req.imageWidth
      : req.imageHeight;
  final lbp = _computeLetterbox(portW, portH);

  // Fill the Float32 input buffer using the combined single-pass loop.
  _fillInputBuffer(req, lbp, inputBuffer);

  // Run TFLite inference.
  interpreter.getInputTensor(0).setTo(inputBuffer);
  interpreter.invoke();

  // Read output tensor raw bytes and reinterpret as float32.
  final rawBytes = interpreter.getOutputTensor(0).data;
  final byteData  = ByteData.sublistView(rawBytes);
  final numFloats = rawBytes.length ~/ 4;
  final outputBuffer = Float32List(numFloats);
  for (int i = 0; i < numFloats; i++) {
    outputBuffer[i] = byteData.getFloat32(i * 4, Endian.little);
  }

  stopwatch.stop();

  final detections = _parseDetections(
    outputBuffer, lbp,
    req.confidenceThreshold,
    req.targetOnly,
    req.isFrontCamera,
  );

  return DetectionResult(
    detections: detections,
    inferenceMs: stopwatch.elapsedMilliseconds,
    lbp: lbp,
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Letterbox helpers
// ════════════════════════════════════════════════════════════════════════════

LetterboxParams _computeLetterbox(int portW, int portH) {
  final scale   = math.min(640.0 / portW, 640.0 / portH);
  final scaledW = (portW * scale).round();
  final scaledH = (portH * scale).round();
  final padX    = (640 - scaledW) / 2.0;
  final padY    = (640 - scaledH) / 2.0;
  return LetterboxParams(
    scale: scale, padX: padX, padY: padY,
    origW: portW, origH: portH,
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Combined YUV → Letterbox inverse → Float32 single-pass converter
// ════════════════════════════════════════════════════════════════════════════

void _fillInputBuffer(FrameRequest req, LetterboxParams lbp, Float32List buf) {
  const double kGray = 114.0 / 255.0;
  buf.fillRange(0, buf.length, kGray);

  final y = req.yBytes;
  final u = req.uBytes;
  final v = req.vBytes;
  final W = req.imageWidth;   // sensor landscape width  (e.g. 1280)
  final H = req.imageHeight;  // sensor landscape height (e.g.  720)

  final double invScale = 1.0 / lbp.scale;

  // Pre-compute lookup tables for pc and pr mapping to eliminate float math, rounding, and bounds checks in inner loops
  final pcTable = Int32List(640);
  final pcPaddingTable = Uint8List(640);
  for (int mc = 0; mc < 640; mc++) {
    final double pc = (mc - lbp.padX) * invScale;
    final int pcRound = pc.round();
    if (pcRound < 0 || pcRound >= lbp.origW) {
      pcPaddingTable[mc] = 1;
    } else {
      pcPaddingTable[mc] = 0;
      pcTable[mc] = pcRound;
    }
  }

  final prTable = Int32List(640);
  final prPaddingTable = Uint8List(640);
  for (int mr = 0; mr < 640; mr++) {
    final double pr = (mr - lbp.padY) * invScale;
    final int prRound = pr.round();
    if (prRound < 0 || prRound >= lbp.origH) {
      prPaddingTable[mr] = 1;
    } else {
      prPaddingTable[mr] = 0;
      prTable[mr] = prRound;
    }
  }

  // Pre-compute row stride offsets
  final yRowOffsets = Int32List(H);
  final uvRowOffsets = Int32List(H);
  final vRowOffsets = Int32List(H);
  for (int i = 0; i < H; i++) {
    yRowOffsets[i] = i * req.yRowStride;
    uvRowOffsets[i] = (i >> 1) * req.uvRowStride;
    vRowOffsets[i] = (i >> 1) * req.vRowStride;
  }

  // Pre-compute pixel stride offsets
  final yPixelOffsets = Int32List(W);
  final uvPixelOffsets = Int32List(W);
  final vPixelOffsets = Int32List(W);
  for (int i = 0; i < W; i++) {
    yPixelOffsets[i] = i * req.yPixelStride;
    uvPixelOffsets[i] = (i >> 1) * req.uvPixelStride;
    vPixelOffsets[i] = (i >> 1) * req.vPixelStride;
  }

  // Pre-compute RGB normalizer table [0..255] -> [0.0..1.0]
  final rgbNormalizer = Float32List(256);
  for (int i = 0; i < 256; i++) {
    rgbNormalizer[i] = i / 255.0;
  }

  // Map 640 coordinates to sensor coordinates depending on orientation
  final rowIndexTable = Int32List(640);
  final colIndexTable = Int32List(640);

  if (req.sensorOrientation == 90) {
    for (int mr = 0; mr < 640; mr++) {
      rowIndexTable[mr] = prTable[mr];
    }
    for (int mc = 0; mc < 640; mc++) {
      colIndexTable[mc] = H - 1 - pcTable[mc];
    }
  } else if (req.sensorOrientation == 270) {
    for (int mr = 0; mr < 640; mr++) {
      rowIndexTable[mr] = W - 1 - prTable[mr];
    }
    for (int mc = 0; mc < 640; mc++) {
      colIndexTable[mc] = pcTable[mc];
    }
  } else if (req.sensorOrientation == 0) {
    for (int mr = 0; mr < 640; mr++) {
      rowIndexTable[mr] = prTable[mr];
    }
    for (int mc = 0; mc < 640; mc++) {
      colIndexTable[mc] = pcTable[mc];
    }
  } else { // 180
    for (int mr = 0; mr < 640; mr++) {
      rowIndexTable[mr] = H - 1 - prTable[mr];
    }
    for (int mc = 0; mc < 640; mc++) {
      colIndexTable[mc] = W - 1 - pcTable[mc];
    }
  }

  final bool isSwap = (req.sensorOrientation == 90 || req.sensorOrientation == 270);
  int outIdx = 0;

  for (int mr = 0; mr < 640; mr++) {
    if (prPaddingTable[mr] == 1) {
      outIdx += 640 * 3;
      continue;
    }
    final int rowVal = rowIndexTable[mr];

    for (int mc = 0; mc < 640; mc++) {
      if (pcPaddingTable[mc] == 1) {
        outIdx += 3;
        continue;
      }
      final int colVal = colIndexTable[mc];

      final int sc = isSwap ? rowVal : colVal;
      final int sr = isSwap ? colVal : rowVal;

      int r, g, b;
      final int uvSc = sc;
      final int uvSr = sr;

      if (req.planeCount >= 3) {
        // ── Android YUV420 (3 separate planes) ──────────────────────────
        final int yIdx  = yRowOffsets[sr] + yPixelOffsets[sc];
        final int uIdx  = uvRowOffsets[uvSr] + uvPixelOffsets[uvSc];
        final int vIdx  = vRowOffsets[uvSr] + vPixelOffsets[uvSc];

        final int yVal = y[yIdx];
        final int uVal = u[uIdx];
        final int vVal = v[vIdx];

        final int c = yVal - 16;
        final int d = uVal - 128;
        final int e = vVal - 128;

        r = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255);
        g = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255);
        b = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255);
      } else if (req.planeCount == 2) {
        // ── iOS semi-planar YUV (Y plane + interleaved UV plane) ─────────
        final int yIdx  = yRowOffsets[sr] + yPixelOffsets[sc];
        final int uvIdx = uvRowOffsets[uvSr] + uvPixelOffsets[uvSc];

        final int yVal = y[yIdx];
        final int uVal = u[uvIdx];
        final int vVal = u[uvIdx + 1];

        final int c = yVal - 16;
        final int d = uVal - 128;
        final int e = vVal - 128;

        r = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255);
        g = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255);
        b = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255);
      } else {
        // ── BGRA (1 plane, iOS fallback) ─────────────────────────────────
        final int idx = yRowOffsets[sr] + yPixelOffsets[sc];
        b = y[idx];
        g = y[idx + 1];
        r = y[idx + 2];
      }

      buf[outIdx++] = rgbNormalizer[r];
      buf[outIdx++] = rgbNormalizer[g];
      buf[outIdx++] = rgbNormalizer[b];
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Detection parsing + reverse letterbox coordinate mapping
// ════════════════════════════════════════════════════════════════════════════

const List<String> _kLabels = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6'];

List<Detection> _parseDetections(
  Float32List output,
  LetterboxParams lbp,
  double threshold,
  bool targetOnly,
  bool isFrontCamera,
) {
  final detections = <Detection>[];
  final int count = output.length ~/ 6;

  for (int i = 0; i < count; i++) {
    final int base = i * 6;
    final double conf = output[base + 4];
    if (conf < threshold) continue;

    final int classId = output[base + 5].round().clamp(0, _kLabels.length - 1);

    // Model box coords are normalised [0..1] relative to the 640×640 input.
    // Convert to model pixels, reverse the letterbox, then normalise to [0..1]
    // in portrait display space.
    double x1 = _unmap(output[base + 0] * 640, lbp.padX, lbp.scale, lbp.origW);
    double y1 = _unmap(output[base + 1] * 640, lbp.padY, lbp.scale, lbp.origH);
    double x2 = _unmap(output[base + 2] * 640, lbp.padX, lbp.scale, lbp.origW);
    double y2 = _unmap(output[base + 3] * 640, lbp.padY, lbp.scale, lbp.origH);

    x1 = x1.clamp(0.0, 1.0);
    y1 = y1.clamp(0.0, 1.0);
    x2 = x2.clamp(0.0, 1.0);
    y2 = y2.clamp(0.0, 1.0);

    // Mirror for front-facing camera.
    if (isFrontCamera) {
      final double tmp = x1;
      x1 = 1.0 - x2;
      x2 = 1.0 - tmp;
    }

    // Centre-reticle filter (already rendered on UI but also enforced here).
    if (targetOnly) {
      final double cx = (x1 + x2) / 2;
      final double cy = (y1 + y2) / 2;
      if (cx < 0.25 || cx > 0.75 || cy < 0.25 || cy > 0.75) continue;
    }

    detections.add(Detection(
      left: x1, top: y1, right: x2, bottom: y2,
      confidence: conf,
      classId: classId,
      label: _kLabels[classId],
    ));
  }

  return detections;
}

/// Inverse letterbox: model pixel → display-normalised coordinate.
///   modelPx  — coordinate in 640×640 model input space (pixels)
///   pad      — padding offset in that dimension (pixels)
///   scale    — uniform letterbox scale factor
///   origSize — original portrait dimension in that axis (pixels)
double _unmap(double modelPx, double pad, double scale, int origSize) {
  return (modelPx - pad) / (scale * origSize);
}

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';

import 'detection_isolate.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: Color(0xFF0F172A),
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );

  List<CameraDescription> cameras = [];
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Failed to get cameras: $e');
  }

  runApp(CariesDetectorApp(cameras: cameras));
}

class CariesDetectorApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const CariesDetectorApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Caries Detector',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0F172A),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF10B981),
          secondary: Color(0xFF3B82F6),
          surface: Color(0xFF1E293B),
        ),
      ),
      home: CariesDetectorHomePage(cameras: cameras),
      debugShowCheckedModeBanner: false,
    );
  }
}

class CariesDetectorHomePage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const CariesDetectorHomePage({super.key, required this.cameras});

  @override
  State<CariesDetectorHomePage> createState() => _CariesDetectorHomePageState();
}

class _CariesDetectorHomePageState extends State<CariesDetectorHomePage>
    with SingleTickerProviderStateMixin {
  // ── Camera ──────────────────────────────────────────────────────────────
  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isFrontCamera = false;
  int _sensorOrientation = 90;

  // ── Detection isolate ────────────────────────────────────────────────────
  Isolate? _detectionIsolate;
  ReceivePort? _resultPort;
  SendPort? _isolateSendPort;
  bool _isIsolateReady = false;
  bool _isolateBusy = false; // true while isolate is processing a frame

  // ── Detection settings ───────────────────────────────────────────────────
  double _confidenceThreshold = 0.35;
  bool _targetOnly = false;
  int _fpsThrottle = 10;

  // ── Runtime stats ────────────────────────────────────────────────────────
  double _fps = 0.0;
  int _lastFrameTime = 0;
  int _inferenceTimeMs = 0;

  // ── Detections (updated whenever isolate posts results) ──────────────────
  List<Detection> _detections = [];

  // ── Scanner animation ────────────────────────────────────────────────────
  late AnimationController _scannerAnimationController;

  // ── ICDAS metadata ───────────────────────────────────────────────────────
  final Map<String, String> _icdasDescriptions = {
    'D0': 'Sound Tooth Structure (No caries or active decay)',
    'D1': 'First Visual Change in Enamel (Visible after air-drying)',
    'D2': 'Distinct Visual Change in Enamel (Visible when wet)',
    'D3': 'Localized Enamel Breakdown (Micro-cavity, no visible dentin)',
    'D4': 'Underlying Dark Shadow from Dentin (Dentin involvement)',
    'D5': 'Distinct Cavity with Visible Dentin (Active cavity)',
    'D6': 'Extensive Distinct Cavity (Deep cavity with visible dentin)',
  };

  // ════════════════════════════════════════════════════════════════════════
  // Lifecycle
  // ════════════════════════════════════════════════════════════════════════

  @override
  void initState() {
    super.initState();
    _scannerAnimationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat(reverse: true);

    _startDetectionIsolate();
    _initCamera();
  }

  @override
  void dispose() {
    // Graceful isolate shutdown
    _isolateSendPort?.send('shutdown');
    _detectionIsolate?.kill(priority: Isolate.immediate);
    _resultPort?.close();

    _cameraController?.dispose();
    _scannerAnimationController.dispose();
    super.dispose();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Isolate management
  // ════════════════════════════════════════════════════════════════════════

  /// Loads the TFLite model bytes from assets and spawns the background
  /// detection isolate.  The isolate sends its own SendPort back as the
  /// first message (handshake), then DetectionResult messages for each frame.
  Future<void> _startDetectionIsolate() async {
    try {
      // Load model bytes on the main isolate (rootBundle unavailable elsewhere)
      final modelData = await rootBundle.load('assets/model.tflite');
      final modelBytes = modelData.buffer.asUint8List();

      debugPrint('Model bytes loaded: ${modelBytes.length} bytes');

      // Open a port for all messages from the isolate
      _resultPort = ReceivePort();
      _resultPort!.listen(_onIsolateMessage);

      // Spawn the isolate
      _detectionIsolate = await Isolate.spawn(
        detectionIsolateEntry,
        IsolateInitPayload(
          modelBytes: modelBytes,
          mainSendPort: _resultPort!.sendPort,
        ),
        debugName: 'DetectionIsolate',
      );
    } catch (e) {
      debugPrint('Failed to start detection isolate: $e');
      _showErrorSnackBar('Failed to load detection model.');
    }
  }

  /// Handles all messages arriving from the detection isolate.
  void _onIsolateMessage(dynamic message) {
    if (message is SendPort) {
      // Handshake: the isolate sent us its SendPort so we can send it frames.
      _isolateSendPort = message;
      if (mounted) setState(() => _isIsolateReady = true);
      debugPrint('Detection isolate ready.');
    } else if (message is DetectionResult) {
      // Inference result — update the UI overlay.
      _isolateBusy = false;

      final now = DateTime.now().millisecondsSinceEpoch;
      final elapsed = now - _lastFrameTime;
      final currentFps = elapsed > 0 ? 1000.0 / elapsed : 0.0;

      if (mounted) {
        setState(() {
          _detections = message.detections;
          _inferenceTimeMs = message.inferenceMs;
          _fps = _fps == 0.0 ? currentFps : (_fps * 0.85 + currentFps * 0.15);
        });
      }
      _lastFrameTime = now;
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  // Camera
  // ════════════════════════════════════════════════════════════════════════

  Future<void> _initCamera() async {
    if (widget.cameras.isEmpty) {
      debugPrint('No cameras available.');
      return;
    }

    final selectedCamera = widget.cameras.firstWhere(
      (c) => _isFrontCamera
          ? c.lensDirection == CameraLensDirection.front
          : c.lensDirection == CameraLensDirection.back,
      orElse: () => widget.cameras.first,
    );

    _cameraController = CameraController(
      selectedCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _cameraController!.initialize();
      await _cameraController!.startImageStream(_onFrameAvailable);
      if (mounted) {
        setState(() {
          _sensorOrientation = selectedCamera.sensorOrientation;
          _isFrontCamera =
              selectedCamera.lensDirection == CameraLensDirection.front;
          _isCameraInitialized = true;
        });
      }
      debugPrint('Camera initialised.');
    } catch (e) {
      debugPrint('Failed to initialise camera: $e');
      _showErrorSnackBar('Camera access denied or failed.');
    }
  }

  void _toggleCameraDirection() async {
    if (_cameraController == null) return;
    await _cameraController!.stopImageStream();
    await _cameraController!.dispose();
    if (mounted) {
      setState(() {
        _isCameraInitialized = false;
        _cameraController = null;
        _isFrontCamera = !_isFrontCamera;
        _detections = [];
      });
    }
    _initCamera();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Frame dispatch — fire-and-forget
  //
  // The camera runs at its native FPS.  When a frame arrives:
  //  • If the isolate is still busy → drop the frame immediately (no queue).
  //  • Otherwise → extract raw bytes, build FrameRequest, send to isolate
  //    (non-blocking), set _isolateBusy = true.
  // The UI thread is never blocked.
  // ════════════════════════════════════════════════════════════════════════
  void _onFrameAvailable(CameraImage image) {
    if (!_isIsolateReady || _isolateBusy) return;

    // Optional: coarse FPS throttle so the isolate isn't fed frames faster
    // than we want to display results, saving some copy work.
    final now = DateTime.now().millisecondsSinceEpoch;
    final minInterval = 1000 ~/ _fpsThrottle;
    if (now - _lastFrameTime < minInterval) return;

    _isolateBusy = true;

    // Copy plane bytes — mandatory for isolate message passing.
    final planes = image.planes;
    final int planeCount = planes.length;

    final Uint8List yBytes = Uint8List.fromList(planes[0].bytes);
    final Uint8List uBytes = planeCount >= 2
        ? Uint8List.fromList(planes[1].bytes)
        : Uint8List(0);
    final Uint8List vBytes = planeCount >= 3
        ? Uint8List.fromList(planes[2].bytes)
        : Uint8List(0);

    final request = FrameRequest(
      yBytes: yBytes,
      uBytes: uBytes,
      vBytes: vBytes,
      imageWidth: image.width,
      imageHeight: image.height,
      yRowStride: planes[0].bytesPerRow,
      yPixelStride: planes[0].bytesPerPixel ?? 1,
      uvRowStride: planeCount >= 2 ? planes[1].bytesPerRow : 0,
      uvPixelStride: planeCount >= 2 ? (planes[1].bytesPerPixel ?? 1) : 0,
      vRowStride: planeCount >= 3 ? planes[2].bytesPerRow : 0,
      vPixelStride: planeCount >= 3 ? (planes[2].bytesPerPixel ?? 1) : 0,
      planeCount: planeCount,
      isFrontCamera: _isFrontCamera,
      sensorOrientation: _sensorOrientation,
      confidenceThreshold: _confidenceThreshold,
      targetOnly: _targetOnly,
    );

    // Send is non-blocking — returns instantly.
    _isolateSendPort!.send(request);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Colour / UI helpers
  // ════════════════════════════════════════════════════════════════════════

  Color _getIcdasColor(int classId) {
    switch (classId) {
      case 0:
        return const Color(0xFF10B981);
      case 1:
        return const Color(0xFF34D399);
      case 2:
        return const Color(0xFF6EE7B7);
      case 3:
        return const Color(0xFFFBBF24);
      case 4:
        return const Color(0xFFF97316);
      case 5:
        return const Color(0xFFEF4444);
      case 6:
        return const Color(0xFFBE123C);
      default:
        return Colors.white;
    }
  }

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: const Color(0xFFBE123C),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _showIcdasInfoSheet() {
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF1E293B),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.white24,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                const Text(
                  'ICDAS Classification Guide',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 4),
                const Text(
                  'International Caries Detection and Assessment System',
                  style: TextStyle(fontSize: 12, color: Colors.white54),
                ),
                const SizedBox(height: 16),
                Expanded(
                  child: ListView.separated(
                    shrinkWrap: true,
                    itemCount: 7,
                    separatorBuilder: (_, __) =>
                        const Divider(color: Colors.white10),
                    itemBuilder: (context, index) {
                      final label = 'D$index';
                      final desc = _icdasDescriptions[label]!;
                      final color = _getIcdasColor(index);
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 4),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 4,
                              ),
                              decoration: BoxDecoration(
                                color: color.withOpacity(0.2),
                                border: Border.all(color: color, width: 1.5),
                                borderRadius: BorderRadius.circular(6),
                              ),
                              child: Text(
                                label,
                                style: TextStyle(
                                  color: color,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 14,
                                ),
                              ),
                            ),
                            const SizedBox(width: 16),
                            Expanded(
                              child: Text(
                                desc,
                                style: const TextStyle(
                                  fontSize: 14,
                                  height: 1.3,
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  // ════════════════════════════════════════════════════════════════════════
  // Build
  // ════════════════════════════════════════════════════════════════════════

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;
    final double cameraViewSize = screenSize.width - 32;
    final bool hasCriticalCaries = _detections.any((d) => d.classId >= 5);

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            // ── Header ──────────────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'ICDAS DETECTOR',
                        style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w900,
                          letterSpacing: 1.5,
                          color: Colors.white,
                        ),
                      ),
                      Row(
                        children: [
                          Container(
                            width: 8,
                            height: 8,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: _isIsolateReady && _isCameraInitialized
                                  ? const Color(0xFF10B981)
                                  : Colors.orange,
                            ),
                          ),
                          const SizedBox(width: 6),
                          Text(
                            _isIsolateReady && _isCameraInitialized
                                ? 'Real-Time Scanner Active'
                                : !_isIsolateReady
                                ? 'Loading AI Model...'
                                : 'Starting Camera...',
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.white54,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  Row(
                    children: [
                      IconButton(
                        onPressed: _showIcdasInfoSheet,
                        icon: const Icon(
                          Icons.info_outline,
                          color: Colors.white70,
                        ),
                        tooltip: 'ICDAS Info',
                      ),
                      IconButton(
                        onPressed: _toggleCameraDirection,
                        icon: const Icon(
                          Icons.flip_camera_android_outlined,
                          color: Colors.white70,
                        ),
                        tooltip: 'Flip Camera',
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // ── Camera + overlay ────────────────────────────────────────
            Expanded(
              child: Center(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Square camera card
                      Container(
                        width: cameraViewSize,
                        height: cameraViewSize,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(24),
                          border: Border.all(
                            color: hasCriticalCaries
                                ? const Color(0xFFEF4444).withOpacity(0.5)
                                : Colors.white10,
                            width: 2,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: hasCriticalCaries
                                  ? const Color(0xFFEF4444).withOpacity(0.1)
                                  : Colors.black45,
                              blurRadius: 20,
                              spreadRadius: 2,
                            ),
                          ],
                        ),
                        clipBehavior: Clip.antiAlias,
                        child: Stack(
                          children: [
                            // 1. Live camera preview (always running smoothly)
                            _isCameraInitialized && _cameraController != null
                                ? Positioned.fill(
                                    child: AspectRatio(
                                      aspectRatio: 1.0,
                                      child: ClipRect(
                                        child: Transform.scale(
                                          scale: _cameraController!
                                              .value
                                              .aspectRatio,
                                          child: Center(
                                            child: CameraPreview(
                                              _cameraController!,
                                            ),
                                          ),
                                        ),
                                      ),
                                    ),
                                  )
                                : Container(
                                    color: const Color(0xFF1E293B),
                                    child: const Center(
                                      child: CircularProgressIndicator(),
                                    ),
                                  ),

                            // 2. Target reticle
                            Positioned.fill(
                              child: CustomPaint(
                                painter: ReticlePainter(
                                  isTargetOnly: _targetOnly,
                                  hasCriticalCaries: hasCriticalCaries,
                                ),
                              ),
                            ),

                            // 3. Scanning line animation
                            if (_isCameraInitialized)
                              AnimatedBuilder(
                                animation: _scannerAnimationController,
                                builder: (context, _) {
                                  return Positioned(
                                    top:
                                        _scannerAnimationController.value *
                                        cameraViewSize,
                                    left: 16,
                                    right: 16,
                                    child: Container(
                                      height: 2,
                                      decoration: BoxDecoration(
                                        boxShadow: [
                                          BoxShadow(
                                            color: hasCriticalCaries
                                                ? const Color(0xFFEF4444)
                                                : const Color(0xFF10B981),
                                            blurRadius: 8,
                                            spreadRadius: 1,
                                          ),
                                        ],
                                        color: hasCriticalCaries
                                            ? const Color(0xFFEF4444)
                                            : const Color(0xFF10B981),
                                      ),
                                    ),
                                  );
                                },
                              ),

                            // 4. Bounding box overlay
                            // Uses RepaintBoundary so only the overlay repaints
                            // when _detections changes, not the whole camera card.
                            Positioned.fill(
                              child: RepaintBoundary(
                                child: LayoutBuilder(
                                  builder: (context, constraints) {
                                    return Stack(
                                      children: _detections.map((d) {
                                        final double left =
                                            d.left * constraints.maxWidth;
                                        final double top =
                                            d.top * constraints.maxHeight;
                                        final double width =
                                            (d.right - d.left) *
                                            constraints.maxWidth;
                                        final double height =
                                            (d.bottom - d.top) *
                                            constraints.maxHeight;
                                        final color = _getIcdasColor(d.classId);

                                        return Positioned(
                                          left: left,
                                          top: top,
                                          width: width,
                                          height: height,
                                          child: Container(
                                            decoration: BoxDecoration(
                                              border: Border.all(
                                                color: color,
                                                width: 2,
                                              ),
                                              borderRadius:
                                                  BorderRadius.circular(8),
                                            ),
                                            child: Stack(
                                              clipBehavior: Clip.none,
                                              children: [
                                                Positioned(
                                                  top: -22,
                                                  left: -2,
                                                  child: Container(
                                                    padding:
                                                        const EdgeInsets.symmetric(
                                                          horizontal: 6,
                                                          vertical: 2,
                                                        ),
                                                    decoration: BoxDecoration(
                                                      color: color,
                                                      borderRadius:
                                                          const BorderRadius.only(
                                                            topLeft:
                                                                Radius.circular(
                                                                  6,
                                                                ),
                                                            topRight:
                                                                Radius.circular(
                                                                  6,
                                                                ),
                                                          ),
                                                    ),
                                                    child: Text(
                                                      '${d.label} ${(d.confidence * 100).toStringAsFixed(0)}%',
                                                      style: const TextStyle(
                                                        color: Colors.black,
                                                        fontSize: 11,
                                                        fontWeight:
                                                            FontWeight.bold,
                                                      ),
                                                    ),
                                                  ),
                                                ),
                                              ],
                                            ),
                                          ),
                                        );
                                      }).toList(),
                                    );
                                  },
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),

                      // ── Live stats card ──────────────────────────────
                      const SizedBox(height: 16),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 12,
                        ),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1E293B),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: Colors.white10),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceAround,
                          children: [
                            _buildInfoStat(
                              'Inference',
                              '${_inferenceTimeMs}ms',
                              Icons.timer_outlined,
                              const Color(0xFF10B981),
                            ),
                            _buildInfoStat(
                              'FPS',
                              _fps.toStringAsFixed(1),
                              Icons.speed_outlined,
                              const Color(0xFF10B981),
                            ),
                            _buildInfoStat(
                              'Detections',
                              '${_detections.length}',
                              Icons.visibility_outlined,
                              hasCriticalCaries
                                  ? Colors.redAccent
                                  : const Color(0xFF10B981),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),

            // ── Controls dashboard ──────────────────────────────────────
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              decoration: const BoxDecoration(
                color: Color(0xFF1E293B),
                borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Confidence slider
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'Sensitivity',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.white70,
                        ),
                      ),
                      Text(
                        '${(_confidenceThreshold * 100).toStringAsFixed(0)}%',
                        style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFF10B981),
                        ),
                      ),
                    ],
                  ),
                  SliderTheme(
                    data: SliderTheme.of(context).copyWith(
                      activeTrackColor: const Color(0xFF10B981),
                      inactiveTrackColor: Colors.white12,
                      thumbColor: const Color(0xFF10B981),
                      overlayColor: const Color(0xFF10B981).withAlpha(32),
                      trackHeight: 4,
                    ),
                    child: Slider(
                      value: _confidenceThreshold,
                      min: 0.15,
                      max: 0.85,
                      onChanged: (val) =>
                          setState(() => _confidenceThreshold = val),
                    ),
                  ),

                  // Option toggles
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Expanded(
                        child: _buildToggleButton(
                          'Center Box Only',
                          _targetOnly,
                          Icons.center_focus_strong_outlined,
                          (val) => setState(() => _targetOnly = val),
                        ),
                      ),
                    ],
                  ),

                  // Critical caries warning
                  if (hasCriticalCaries) ...[
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: const Color(0xFFBE123C).withOpacity(0.15),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: const Color(0xFFBE123C),
                          width: 1,
                        ),
                      ),
                      child: Row(
                        children: [
                          const Icon(
                            Icons.warning_amber_rounded,
                            color: Color(0xFFEF4444),
                            size: 22,
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  'SEVERE CARIES DETECTED (D5/D6)',
                                  style: TextStyle(
                                    color: Color(0xFFEF4444),
                                    fontWeight: FontWeight.bold,
                                    fontSize: 12,
                                    letterSpacing: 0.5,
                                  ),
                                ),
                                Text(
                                  'Distinct or extensive cavitations identified. Consult a dentist.',
                                  style: TextStyle(
                                    color: Colors.white.withOpacity(0.8),
                                    fontSize: 11,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════════
  // Widget helpers
  // ════════════════════════════════════════════════════════════════════════

  Widget _buildInfoStat(
    String label,
    String value,
    IconData icon,
    Color color,
  ) {
    return Column(
      children: [
        Icon(icon, size: 20, color: color),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.bold,
            letterSpacing: 0.5,
          ),
        ),
        Text(
          label,
          style: const TextStyle(fontSize: 10, color: Colors.white38),
        ),
      ],
    );
  }

  Widget _buildToggleButton(
    String label,
    bool value,
    IconData icon,
    Function(bool) onChanged,
  ) {
    return InkWell(
      onTap: () => onChanged(!value),
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: value
              ? const Color(0xFF10B981).withOpacity(0.1)
              : const Color(0xFF0F172A),
          border: Border.all(
            color: value ? const Color(0xFF10B981) : Colors.white10,
            width: 1.5,
          ),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 18,
              color: value ? const Color(0xFF10B981) : Colors.white60,
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                label,
                overflow: TextOverflow.ellipsis,
                maxLines: 1,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: value ? FontWeight.bold : FontWeight.normal,
                  color: value ? const Color(0xFF10B981) : Colors.white70,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Reticle painter
// ════════════════════════════════════════════════════════════════════════════

class ReticlePainter extends CustomPainter {
  final bool isTargetOnly;
  final bool hasCriticalCaries;

  ReticlePainter({required this.isTargetOnly, required this.hasCriticalCaries});

  @override
  void paint(Canvas canvas, Size size) {
    final double w = size.width;
    final double h = size.height;

    final double targetW = w * 0.50;
    final double targetH = h * 0.50;
    final double left = (w - targetW) / 2;
    final double top = (h - targetH) / 2;
    final double right = left + targetW;
    final double bottom = top + targetH;

    final Paint paint = Paint()
      ..color = isTargetOnly
          ? (hasCriticalCaries
                ? const Color(0xFFEF4444)
                : const Color(0xFF10B981))
          : Colors.white24
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    const double len = 24.0;

    // Top-Left
    canvas.drawLine(Offset(left, top), Offset(left + len, top), paint);
    canvas.drawLine(Offset(left, top), Offset(left, top + len), paint);
    // Top-Right
    canvas.drawLine(Offset(right, top), Offset(right - len, top), paint);
    canvas.drawLine(Offset(right, top), Offset(right, top + len), paint);
    // Bottom-Left
    canvas.drawLine(Offset(left, bottom), Offset(left + len, bottom), paint);
    canvas.drawLine(Offset(left, bottom), Offset(left, bottom - len), paint);
    // Bottom-Right
    canvas.drawLine(Offset(right, bottom), Offset(right - len, bottom), paint);
    canvas.drawLine(Offset(right, bottom), Offset(right, bottom - len), paint);

    if (isTargetOnly) {
      final Paint bg = Paint()..color = Colors.black.withOpacity(0.3);
      canvas.drawRect(Rect.fromLTRB(0, 0, w, top), bg);
      canvas.drawRect(Rect.fromLTRB(0, bottom, w, h), bg);
      canvas.drawRect(Rect.fromLTRB(0, top, left, bottom), bg);
      canvas.drawRect(Rect.fromLTRB(right, top, w, bottom), bg);
    }
  }

  @override
  bool shouldRepaint(covariant ReticlePainter old) =>
      old.isTargetOnly != isTargetOnly ||
      old.hasCriticalCaries != hasCriticalCaries;
}

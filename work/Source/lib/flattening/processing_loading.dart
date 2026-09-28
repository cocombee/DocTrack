import 'dart:async';

import 'package:flutter/material.dart';

/// One loading surface across preparation and rendering, with a shared start
/// time so changing stages cannot hide how long the image has been processing.
class ImageProcessingLoading extends StatefulWidget {
  const ImageProcessingLoading({
    super.key,
    required this.startedAt,
    required this.stage,
    this.onClose,
  });

  final DateTime startedAt;
  final String stage;
  final VoidCallback? onClose;

  @override
  State<ImageProcessingLoading> createState() => _ImageProcessingLoadingState();
}

class _ImageProcessingLoadingState extends State<ImageProcessingLoading> {
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final seconds = DateTime.now().difference(widget.startedAt).inSeconds;
    final elapsed = Duration(seconds: seconds < 0 ? 0 : seconds);
    final takingLonger = elapsed >= const Duration(seconds: 12);
    final time = elapsed.inMinutes == 0
        ? '${elapsed.inSeconds}s elapsed'
        : '${elapsed.inMinutes}m ${elapsed.inSeconds % 60}s elapsed';
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 360),
          child: DecoratedBox(
            decoration: BoxDecoration(
              color: const Color(0xFF111113),
              border: Border.all(color: const Color(0xFF27272A)),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Stack(
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(32, 32, 32, 24),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Color(0xFFFAFAFA),
                        ),
                      ),
                      const SizedBox(height: 16),
                      Semantics(
                        liveRegion: true,
                        child: Text(
                          takingLonger
                              ? 'This image is taking longer to straighten'
                              : 'Analyzing image…',
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            color: Color(0xFFFAFAFA),
                            fontSize: 13,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        widget.stage,
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          color: Color(0xFFA1A1AA),
                          fontSize: 11,
                        ),
                      ),
                      const SizedBox(height: 12),
                      ExcludeSemantics(
                        // Avoid announcing every elapsed-time tick.
                        child: Text(
                          time,
                          style: const TextStyle(
                            color: Color(0xFFA1A1AA),
                            fontSize: 11,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                if (widget.onClose != null)
                  Positioned(
                    top: 4,
                    right: 4,
                    child: IconButton(
                      onPressed: widget.onClose,
                      tooltip: 'Close preview',
                      icon: const Icon(Icons.close, size: 16),
                      color: const Color(0xFFA1A1AA),
                      constraints: const BoxConstraints.tightFor(
                        width: 28,
                        height: 28,
                      ),
                      padding: EdgeInsets.zero,
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

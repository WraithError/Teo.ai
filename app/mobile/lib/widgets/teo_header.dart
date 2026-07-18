import 'package:flutter/material.dart';
import 'package:teo_ai/main.dart';

class TeoHeader extends StatelessWidget {
  final bool online;
  final bool modelLoaded;

  const TeoHeader({
    super.key,
    required this.online,
    required this.modelLoaded,
  });

  @override
  Widget build(BuildContext context) {
    final statusLabel = !online
        ? 'OFFLINE'
        : modelLoaded
            ? 'MODEL LIVE'
            : 'PLACEHOLDER';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: TeoApp.border)),
      ),
      child: Row(
        children: [
          _BarcodeLogo(),
          const SizedBox(width: 10),
          const Text(
            'TEO',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              letterSpacing: 4,
              color: TeoApp.dimWhite,
              fontFamily: 'monospace',
            ),
          ),
          const SizedBox(width: 6),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
            decoration: BoxDecoration(
              border: Border.all(color: TeoApp.yellow),
            ),
            child: const Text(
              '.AI',
              style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.bold,
                color: TeoApp.yellow,
                letterSpacing: 2,
                fontFamily: 'monospace',
              ),
            ),
          ),
          const SizedBox(width: 8),
          const Text(
            'v0.4',
            style: TextStyle(
              fontSize: 10,
              color: TeoApp.muted,
              letterSpacing: 2,
              fontFamily: 'monospace',
            ),
          ),
          const Spacer(),
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: online ? const Color(0xFF22C55E) : TeoApp.muted,
              boxShadow: online
                  ? [const BoxShadow(color: Color(0x8822C55E), blurRadius: 6)]
                  : null,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            statusLabel,
            style: const TextStyle(
              fontSize: 10,
              color: TeoApp.muted,
              letterSpacing: 2,
              fontFamily: 'monospace',
            ),
          ),
        ],
      ),
    );
  }
}

class _BarcodeLogo extends StatelessWidget {
  static const heights = [28.0, 20, 26, 16, 28, 22, 18, 28, 14, 24, 28, 20];

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 28,
      child: Stack(
        alignment: Alignment.center,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              for (final h in heights)
                Container(
                  width: 2,
                  height: h,
                  margin: const EdgeInsets.only(right: 2),
                  color: TeoApp.dimWhite,
                ),
            ],
          ),
          Transform.rotate(
            angle: 0.21,
            child: Container(
              width: 2,
              height: 28,
              color: TeoApp.yellow,
            ),
          ),
        ],
      ),
    );
  }
}

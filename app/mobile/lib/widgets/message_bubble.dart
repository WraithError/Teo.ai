import 'package:flutter/material.dart';
import 'package:teo_ai/main.dart';

class MessageBubble extends StatelessWidget {
  final String role;
  final String content;
  final bool thinking;

  const MessageBubble({
    super.key,
    required this.role,
    required this.content,
    this.thinking = false,
  });

  @override
  Widget build(BuildContext context) {
    final isUser = role == 'user';

    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Align(
        alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
        child: Container(
          constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width * 0.72,
          ),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: isUser ? TeoApp.surface : null,
            border: isUser
                ? Border.all(color: TeoApp.border)
                : const Border(
                    left: BorderSide(color: TeoApp.yellow, width: 2),
                  ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                isUser ? 'YOU' : 'TEO',
                style: TextStyle(
                  fontSize: 10,
                  letterSpacing: 2,
                  color: isUser ? TeoApp.muted : TeoApp.yellow,
                  fontFamily: 'monospace',
                ),
              ),
              const SizedBox(height: 6),
              Text(
                content,
                style: TextStyle(
                  fontSize: 13,
                  height: 1.6,
                  color: thinking ? TeoApp.muted : TeoApp.dimWhite,
                  fontFamily: 'monospace',
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:teo_ai/screens/chat_screen.dart';

void main() {
  runApp(const TeoApp());
}

class TeoApp extends StatelessWidget {
  const TeoApp({super.key});

  static const black      = Color(0xFF0A0A0A);
  static const surface    = Color(0xFF111111);
  static const border     = Color(0xFF222222);
  static const dimWhite   = Color(0xFFC8C8C8);
  static const muted      = Color(0xFF555555);
  static const yellow     = Color(0xFFF5C400);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TEO.AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: black,
        fontFamily: 'monospace',
        colorScheme: const ColorScheme.dark(
          primary: yellow,
          surface: surface,
          onSurface: dimWhite,
        ),
      ),
      home: const ChatScreen(),
    );
  }
}

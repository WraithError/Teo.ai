import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:teo_ai/main.dart';
import 'package:teo_ai/widgets/teo_header.dart';
import 'package:teo_ai/widgets/message_bubble.dart';

class ChatMessage {
  final String role;
  final String content;
  const ChatMessage({required this.role, required this.content});

  Map<String, String> toJson() => {'role': role, 'content': content};
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  // Change this to your machine IP when testing on a physical device.
  // Android emulator: use 10.0.2.2:8000
  static const apiBase = String.fromEnvironment(
    'TEO_API',
    defaultValue: 'http://10.0.2.2:8000',
  );

  final _controller = TextEditingController();
  final _scroll = ScrollController();
  final List<ChatMessage> _messages = [];
  final List<Map<String, String>> _history = [];

  bool _sending = false;
  bool _online = false;
  bool _modelLoaded = false;

  @override
  void initState() {
    super.initState();
    _messages.add(const ChatMessage(
      role: 'teo',
      content: 'Accepted. DEPLOY. DEFEND. EVOLVE. — Talk to Teo.',
    ));
    _checkHealth();
  }

  @override
  void dispose() {
    _controller.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _checkHealth() async {
    try {
      final res = await http
          .get(Uri.parse('$apiBase/api/health'))
          .timeout(const Duration(seconds: 5));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        setState(() {
          _online = true;
          _modelLoaded = data['model_loaded'] == true;
        });
        return;
      }
    } catch (_) {}
    setState(() {
      _online = false;
      _modelLoaded = false;
    });
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _sending) return;

    setState(() {
      _sending = true;
      _messages.add(ChatMessage(role: 'user', content: text));
      _history.add({'role': 'user', 'content': text});
      if (_history.length > 20) _history.removeAt(0);
      _controller.clear();
    });
    _scrollToBottom();

    try {
      final res = await http
          .post(
            Uri.parse('$apiBase/api/chat'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'message': text, 'history': _history}),
          )
          .timeout(const Duration(seconds: 30));

      String reply;
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        reply = data['response'] as String;
        _history.add({'role': 'teo', 'content': reply});
        if (_history.length > 20) _history.removeAt(0);
        await _checkHealth();
      } else {
        reply = '[server error ${res.statusCode}]';
      }

      setState(() => _messages.add(ChatMessage(role: 'teo', content: reply)));
    } catch (_) {
      setState(() {
        _online = false;
        _messages.add(const ChatMessage(
          role: 'teo',
          content: '[connection failed — start backend on $apiBase]',
        ));
      });
    }

    setState(() => _sending = false);
    _scrollToBottom();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            TeoHeader(
              online: _online,
              modelLoaded: _modelLoaded,
            ),
            Expanded(
              child: ListView.builder(
                controller: _scroll,
                padding: const EdgeInsets.all(20),
                itemCount: _messages.length + (_sending ? 1 : 0),
                itemBuilder: (context, i) {
                  if (i == _messages.length) {
                    return const MessageBubble(
                      role: 'teo',
                      content: 'TEO IS THINKING...',
                      thinking: true,
                    );
                  }
                  final m = _messages[i];
                  return MessageBubble(role: m.role, content: m.content);
                },
              ),
            ),
            Container(
              decoration: const BoxDecoration(
                border: Border(top: BorderSide(color: TeoApp.border)),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      style: const TextStyle(
                        color: TeoApp.dimWhite,
                        fontSize: 13,
                        fontFamily: 'monospace',
                      ),
                      decoration: const InputDecoration(
                        hintText: 'Talk to Teo...',
                        hintStyle: TextStyle(color: TeoApp.muted),
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 16,
                        ),
                        filled: true,
                        fillColor: TeoApp.surface,
                      ),
                      onSubmitted: (_) => _send(),
                    ),
                  ),
                  Material(
                    color: _sending ? const Color(0xFFA38200) : TeoApp.yellow,
                    child: InkWell(
                      onTap: _sending ? null : _send,
                      child: const Padding(
                        padding: EdgeInsets.symmetric(
                          horizontal: 24,
                          vertical: 18,
                        ),
                        child: Text(
                          'SEND',
                          style: TextStyle(
                            color: TeoApp.black,
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 2,
                            fontFamily: 'monospace',
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

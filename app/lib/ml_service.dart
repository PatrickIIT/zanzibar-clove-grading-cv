import 'dart:io';
import 'dart:math';
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'image_processor.dart';
import 'package:ubora_ai/l10n/app_localizations.dart'; // ← only new import

class MLService {
  // ── Single clove (TFLite YOLOv8-seg) ─────────────────────────────────────
  static Interpreter? _interpreter;
  static bool _isInitialized = false;
  static bool _isProcessing = false;

  // ── Batch clove (EfficientNet-Lite0 INT8 TFLite) ─────────────────────────
  static Interpreter? _batchInterpreter;
  static bool _batchInitialized = false;

  // ── Batch model constants ─────────────────────────────────────────────────
  static const int _batchImgSize = 224;
  static const List<String> _batchClasses = [
    'Grade I', 'Grade II', 'Grade III', 'Grade IV', 'Not_Clove'
  ];

  // ── Single clove model constants (DO NOT CHANGE) ──────────────────────────
  static const int outputDetections = 1029;
  static const int outputAttributes = 41;
  static const int inputSize = 224;
  static const int numClasses = 4;
  static const double minAspectRatio = 0.5;
  static const double maxAspectRatio = 4.0;
  static const double minAreaRatio = 0.002;
  static const double maxAreaRatio = 0.85;
  static const double confidenceThreshold = 0.25;
  static const double minCloveConfidence = 0.40;
  static const double nmsThreshold = 0.45;
  static const double gradeIMpetaMax = 3.0;
  static const double gradeIIMpetaMax = 7.0;
  static const double gradeIIIMpetaMax = 20.0;

  static int getInputSize(bool isBatchMode) => isBatchMode ? 640 : 224;

  // ══════════════════════════════════════════════════════════════════════════
  // INITIALISATION
  // ══════════════════════════════════════════════════════════════════════════

  static Future<void> initialize() async {
    await Future.wait([_initTflite(), _initBatchTflite()]);
  }

  // Original single-clove TFLite init — unchanged
  static Future<void> _initTflite() async {
    if (_isInitialized) {
      print('✅ Model already initialized');
      return;
    }
    try {
      print("Loading YOLOv8-seg model from assets/best_int8.tflite...");
      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset(
        'assets/best_int8.tflite',
        options: options,
      );
      _isInitialized = true;
      print('✅ YOLOv8-seg model loaded successfully');
      final inputShape = _interpreter!.getInputTensor(0).shape;
      print('Input shape: $inputShape');
      final outputTensors = _interpreter!.getOutputTensors();
      for (int i = 0; i < outputTensors.length; i++) {
        print('Output $i shape: ${outputTensors[i].shape}');
      }
    } catch (e, stack) {
      _isInitialized = false;
      print('❌ Failed to load model: $e');
      print('Stack trace: $stack');
    }
  }

  // New batch EfficientNet-Lite0 INT8 TFLite init
  static Future<void> _initBatchTflite() async {
    if (_batchInitialized) return;
    try {
      print("Loading EfficientNet-Lite0 INT8 batch model from assets/clove_grader_int8.tflite...");
      final options = InterpreterOptions()..threads = 4;
      _batchInterpreter = await Interpreter.fromAsset(
        'assets/clove_grader_int8.tflite',
        options: options,
      );
      _batchInitialized = true;
      print('✅ EfficientNet-Lite0 batch model loaded successfully');
      final inputShape  = _batchInterpreter!.getInputTensor(0).shape;
      final outputShape = _batchInterpreter!.getOutputTensor(0).shape;
      print('Batch input shape:  $inputShape');
      print('Batch output shape: $outputShape');

      final inputType = _batchInterpreter!.getInputTensor(0).type;
      _batchIsInt8 = (inputType.toString().toLowerCase().contains('uint8'));
      print('Batch model input type: ${inputType} → isInt8=$_batchIsInt8');
    } catch (e, stack) {
      _batchInitialized = false;
      print('❌ Failed to load batch model: $e');
      print('Stack trace: $stack');
    }
  }

  static bool _batchIsInt8 = true;

  // ══════════════════════════════════════════════════════════════════════════
  // PUBLIC ENTRY POINT
  // ══════════════════════════════════════════════════════════════════════════

  static Future<Map<String, dynamic>> analyzeImage(File imageFile, {bool isBatchMode = false}) async {
    if (isBatchMode) {
      return _analyzeBatch(imageFile);
    }

    if (_isProcessing) {
      return {
        'isCloveDetected': false,
        'error': AppStrings.current.mlErrorBusy,               // ← was hardcoded Swahili
      };
    }

    _isProcessing = true;

    try {
      final optimizedFile = await ImageProcessor.optimizeImageForModel(
        imageFile,
        isBatchMode: isBatchMode
      );

      final bytes = await optimizedFile.readAsBytes();
      final image = img.decodeImage(bytes);
      if (image == null) throw Exception(AppStrings.current.mlErrorDecodeImage);

      final resized = ImageProcessor.resizeForModel(image);

      final detectionResult = await _getDetections(
        resized,
        image.width,
        image.height,
        isBatchMode: isBatchMode
      );

      await optimizedFile.delete();

      if (!detectionResult['isValid']) {
        _isProcessing = false;
        return {
          'grade': '',
          'confidence': 0.0,
          'isCloveDetected': false,
          'details': {},
          'error': detectionResult['reason'],
        };
      }

      final detections = detectionResult['detections'] as List<Map<String, dynamic>>;

      final result = await _processSingleGrading(detections, resized);

      _isProcessing = false;
      return result;

    } catch (e) {
      print('Analysis error: $e');
      _isProcessing = false;
      return {
        'grade': '',
        'confidence': 0.0,
        'isCloveDetected': false,
        'details': {},
        'error': AppStrings.current.mlErrorProcessing,          // ← was hardcoded Swahili
      };
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // SINGLE CLOVE — logic unchanged; only string literals replaced
  // ══════════════════════════════════════════════════════════════════════════

  static Future<Map<String, dynamic>> _getDetections(
      img.Image image,
      int origWidth,
      int origHeight,
      {bool isBatchMode = false}) async {

    if (!_isInitialized) {
      return {
        'isValid': false,
        'reason': AppStrings.current.mlErrorNotInitialized,     // ← was hardcoded Swahili
        'detections': []
      };
    }

    try {
      print('Starting detection on ${origWidth}x${origHeight} image ${isBatchMode ? "(Batch Mode)" : "(Single Mode)"}');

      final input = _preprocessImage(image);
      final output = await _runInference(input);
      final allDetections = _parseYoloOutput(output[0], origWidth, origHeight);
      print('Total raw detections: ${allDetections.length}');

      final double effectiveMinConfidence = isBatchMode ? 0.40 : 0.60;
      final validDetections = allDetections.where((d) =>
        d['confidence'] >= effectiveMinConfidence
      ).toList();

      print('Valid detections (conf >= $effectiveMinConfidence): ${validDetections.length}');

      if (validDetections.isEmpty) {
        if (allDetections.isNotEmpty) {
          print('Low confidence detections:');
          for (var d in allDetections.take(5)) {
            print('  ${d['grade']} conf=${d['confidence'].toStringAsFixed(4)}');
          }
        }
        return {
          'isValid': false,
          'reason': AppStrings.current.mlErrorLowConfidence,    // ← was hardcoded Swahili
          'detections': []
        };
      }

      final bestDetection = validDetections.first;
      final bbox = bestDetection['bbox'] as List<double>;
      final objWidth = bbox[2] - bbox[0];
      final objHeight = bbox[3] - bbox[1];
      final area = objWidth * objHeight;
      final imageArea = origWidth * origHeight.toDouble();
      final areaRatio = area / imageArea;

      print('Best detection: ${bestDetection['grade']}, conf=${bestDetection['confidence']}');
      print('Area ratio: ${areaRatio.toStringAsFixed(4)}');

      if (isBatchMode) {
        if (areaRatio < 0.003) {
          return {
            'isValid': false,
            'reason': AppStrings.current.mlErrorBatchTooSmall,  // ← was hardcoded Swahili
            'detections': []
          };
        }
        if (areaRatio > 0.92) {
          return {
            'isValid': false,
            'reason': AppStrings.current.mlErrorBatchTooClose,  // ← was hardcoded Swahili
            'detections': []
          };
        }
      } else {
        if (areaRatio < 0.015) {
          return {
            'isValid': false,
            'reason': AppStrings.current.mlErrorTooFar,         // ← was hardcoded Swahili
            'detections': []
          };
        }
        if (areaRatio > 0.90) {
          return {
            'isValid': false,
            'reason': AppStrings.current.mlErrorTooClose,       // ← was hardcoded Swahili
            'detections': []
          };
        }
      }

      print('✅ Validation passed → ${bestDetection['grade']} accepted');

      return {
        'isValid': true,
        'detections': allDetections,
        'reason': null,
      };

    } catch (e) {
      print('Detection error: $e');
      return {
        'isValid': false,
        'reason': AppStrings.current.mlErrorProcessing,         // ← was hardcoded Swahili
        'detections': []
      };
    }
  }

  static Future<List<dynamic>> _runInference(List input) async {
    if (_interpreter == null || !_isInitialized) {
      await _initTflite();
    }

    final runner = _interpreter;
    if (runner == null) throw Exception("Interpreter is null");

    var output0 = List.filled(1 * 41 * 1029, 0.0).reshape([1, 41, 1029]);
    var output1 = List.filled(1 * 56 * 56 * 32, 0.0).reshape([1, 56, 56, 32]);

    final outputs = {0: output0, 1: output1};

    print("Running multi-output inference...");
    runner.runForMultipleInputs([input], outputs);

    print('After inference - output0 runtimeType: ${output0.runtimeType}');
    print('After inference - output0 length: ${output0.length}');
    if (output0.isNotEmpty && output0[0] is List) {
      print('output0[0] length: ${(output0[0] as List).length}');
    }

    return [output0, output1];
  }

  static List<Map<String, dynamic>> _parseYoloOutput(
      List<dynamic> outputs, int origWidth, int origHeight) {

    final detections = <Map<String, dynamic>>[];

    if (outputs.isEmpty || outputs[0] == null) {
      print('Outputs list is empty');
      return detections;
    }

    try {
      final dynamic rawOutput0 = outputs[0];

      print('Output0 runtimeType: ${rawOutput0.runtimeType}');
      print('Output0 length (rows): ${rawOutput0.length}');

      final List<List<double>> dataMatrix;

      if (rawOutput0 is List<List<double>>) {
        dataMatrix = rawOutput0;
      } else if (rawOutput0 is List) {
        dataMatrix = List<List<double>>.from(
          rawOutput0.map((row) => List<double>.from(row as List))
        );
      } else {
        print('ERROR: Unexpected output type: ${rawOutput0.runtimeType}');
        return detections;
      }

      print('✅ Parsed Matrix: ${dataMatrix.length} rows × ${dataMatrix[0].length} columns');

      if (dataMatrix.length < 5) {
        print('ERROR: Not enough attributes (rows)');
        return detections;
      }

      print('Parsing ${outputDetections} detections...');

      for (int i = 0; i < outputDetections; i++) {
        try {
          final double cx = _safeGetValue(dataMatrix[0], i);
          final double cy = _safeGetValue(dataMatrix[1], i);
          final double w  = _safeGetValue(dataMatrix[2], i);
          final double h  = _safeGetValue(dataMatrix[3], i);

          if (w <= 0.01 || h <= 0.01) continue;

          double maxScore = 0.0;
          int bestClass = 0;

          for (int c = 0; c < numClasses; c++) {
            final int classIndex = 4 + c;
            if (classIndex >= dataMatrix.length) break;
            final double score = _safeGetValue(dataMatrix[classIndex], i);
            if (score > maxScore) {
              maxScore = score;
              bestClass = c;
            }
          }

          if (maxScore > 0.25) {
            final double x1 = ((cx - w / 2) * origWidth).clamp(0.0, origWidth.toDouble());
            final double y1 = ((cy - h / 2) * origHeight).clamp(0.0, origHeight.toDouble());
            final double x2 = ((cx + w / 2) * origWidth).clamp(0.0, origWidth.toDouble());
            final double y2 = ((cy + h / 2) * origHeight).clamp(0.0, origHeight.toDouble());

            if (x2 > x1 && y2 > y1) {
              final grades = ['Grade I', 'Grade II', 'Grade III', 'Grade IV'];
              detections.add({
                'bbox': [x1, y1, x2, y2],
                'grade': grades[bestClass],
                'confidence': maxScore,
                'classId': bestClass,
              });
              if (detections.length <= 12) {
                print('✅ Detection ${detections.length}: ${grades[bestClass]} conf=${maxScore.toStringAsFixed(4)}');
              }
            }
          }
        } catch (e) {
          continue;
        }
      }
    } catch (e) {
      print('Critical parsing error: $e');
    }

    print('Raw detections before NMS: ${detections.length}');

    if (detections.isEmpty) {
      print('⚠️ Still 0 detections. The model may be outputting very low scores or the post-processing needs further adjustment.');
      return detections;
    }

    final finalDetections = _applyNMS(detections);
    print('Detections after NMS: ${finalDetections.length}');
    return finalDetections;
  }

  static double _safeGetValue(List<dynamic> list, int index) {
    if (index < 0 || index >= list.length) return 0.0;
    final value = list[index];
    if (value is double) return value;
    if (value is int) return value.toDouble();
    if (value is num) return value.toDouble();
    try {
      return double.tryParse(value.toString()) ?? 0.0;
    } catch (_) {
      return 0.0;
    }
  }

  static List<Map<String, dynamic>> _applyNMS(List<Map<String, dynamic>> detections) {
    if (detections.isEmpty) return [];
    detections.sort((a, b) => b['confidence'].compareTo(a['confidence']));
    final kept = <Map<String, dynamic>>[];
    final suppressed = List.filled(detections.length, false);
    for (int i = 0; i < detections.length; i++) {
      if (suppressed[i]) continue;
      kept.add(detections[i]);
      for (int j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;
        final iou = _calculateIoU(detections[i]['bbox'], detections[j]['bbox']);
        if (iou > nmsThreshold) suppressed[j] = true;
      }
    }
    return kept;
  }

  static double _calculateIoU(List<double> box1, List<double> box2) {
    final x1 = max(box1[0], box2[0]);
    final y1 = max(box1[1], box2[1]);
    final x2 = min(box1[2], box2[2]);
    final y2 = min(box1[3], box2[3]);
    if (x2 <= x1 || y2 <= y1) return 0.0;
    final intersection = (x2 - x1) * (y2 - y1);
    final area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    final area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    final union = area1 + area2 - intersection;
    return intersection / union;
  }

  static Future<Map<String, dynamic>> _processSingleGrading(
      List<Map<String, dynamic>> detections, img.Image image) async {
    if (detections.isEmpty) {
      return {
        'isCloveDetected': false,
        'error': AppStrings.current.mlErrorNoClove,             // ← was hardcoded Swahili
      };
    }

    final best    = detections.first;
    final s       = AppStrings.current;
    final color   = _analyzeColor(image);
    final size    = _analyzeSize(image);
    final defects = _detectDefects(image);

    final auditTrail = s.singleAuditTrail(
      grade:      best['grade'],
      confidence: (best['confidence'] * 100).toStringAsFixed(1),
      color:      color,
      size:       size,
      defects:    defects,
    );

    return {
      'isCloveDetected': true,
      'grade': best['grade'],
      'confidence': best['confidence'],
      'details': {
        'color': color,
        'size': size,
        'defects': defects,
      },
      'auditTrail': auditTrail,
      'error': null,
    };
  }

  static List _preprocessImage(img.Image image) {
    final resizedImage = img.copyResize(image, width: inputSize, height: inputSize);
    var input = List.generate(1, (_) =>
      List.generate(inputSize, (_) =>
        List.generate(inputSize, (_) =>
          List.filled(3, 0.0)
        )
      )
    );
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resizedImage.getPixel(x, y);
        input[0][y][x][0] = pixel.r / 255.0;
        input[0][y][x][1] = pixel.g / 255.0;
        input[0][y][x][2] = pixel.b / 255.0;
      }
    }
    return input;
  }

  // Color, size, defect helpers — labels now come from AppStrings
  static String _analyzeColor(img.Image image) {
    int rSum = 0, gSum = 0, bSum = 0;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        rSum += pixel.r.toInt();
        gSum += pixel.g.toInt();
        bSum += pixel.b.toInt();
      }
    }
    final total = image.width * image.height;
    final avgR = rSum / total;
    final avgG = gSum / total;
    final avgB = bSum / total;
    final s = AppStrings.current;
    if (avgR > 160 && avgG > 100 && avgB < 110) return s.colorGolden;
    if (avgR > 120 && avgG > 70  && avgB < 100) return s.colorLight;
    if (avgR > 80  && avgG > 50  && avgB < 80)  return s.colorBrown;
    if (avgR < 80  && avgG < 60  && avgB < 70)  return s.colorBlack;
    return s.colorMixed;
  }

  static String _analyzeSize(img.Image image) {
    final area = image.width * image.height;
    final s = AppStrings.current;
    if (area > 40000) return s.sizeLarge;
    if (area > 25000) return s.sizeMedium;
    if (area > 15000) return s.sizeSmall;
    return s.sizeVerySmall;
  }

  static String _detectDefects(img.Image image) {
    int darkSpots = 0;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        if (pixel.r.toInt() < 50 && pixel.g.toInt() < 50 && pixel.b.toInt() < 50) {
          darkSpots++;
        }
      }
    }
    final total = image.width * image.height;
    final defectPercent = darkSpots / total * 100;
    final s = AppStrings.current;
    if (defectPercent < 5)  return s.defectsFew;
    if (defectPercent < 12) return s.defectsMedium;
    if (defectPercent < 20) return s.defectsMany;
    return s.defectsVeryMany;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // BATCH CLOVE — EfficientNet-Lite0 INT8 TFLite
  // Logic unchanged; only string literals replaced with AppStrings.current.*
  // ══════════════════════════════════════════════════════════════════════════

  static Future<Map<String, dynamic>> _analyzeBatch(File imageFile) async {
    if (!_batchInitialized) {
      await _initBatchTflite();
      if (!_batchInitialized) {
        return {
          'isCloveDetected': false,
          'grade': '',
          'confidence': 0.0,
          'details': {},
          'error': AppStrings.current.mlErrorModelNotLoaded,    // ← was hardcoded Swahili
        };
      }
    }

    try {
      final rawBytes = await imageFile.readAsBytes();
      final image = img.decodeImage(rawBytes);
      if (image == null) throw Exception(AppStrings.current.mlErrorDecodeImage);

      print('Batch: original image ${image.width}x${image.height}');

      img.Image working = image;

      const int maxEdge = 640;
      if (working.width > maxEdge || working.height > maxEdge) {
        if (working.width >= working.height) {
          working = img.copyResize(working, width: maxEdge,
              interpolation: img.Interpolation.average);
        } else {
          working = img.copyResize(working, height: maxEdge,
              interpolation: img.Interpolation.average);
        }
      }

      final sq = working.width < working.height ? working.width : working.height;
      final cropX = (working.width  - sq) ~/ 2;
      final cropY = (working.height - sq) ~/ 2;
      working = img.copyCrop(working, x: cropX, y: cropY, width: sq, height: sq);

      final resized = img.copyResize(working,
          width: _batchImgSize, height: _batchImgSize,
          interpolation: img.Interpolation.average);
      print('Batch: resized to ${resized.width}x${resized.height}');

      List inputTensor;
      if (_batchIsInt8) {
        final flat = Uint8List(_batchImgSize * _batchImgSize * 3);
        int idx = 0;
        for (int y = 0; y < _batchImgSize; y++) {
          for (int x = 0; x < _batchImgSize; x++) {
            final p = resized.getPixel(x, y);
            flat[idx++] = p.r.toInt().clamp(0, 255);
            flat[idx++] = p.g.toInt().clamp(0, 255);
            flat[idx++] = p.b.toInt().clamp(0, 255);
          }
        }
        inputTensor = [flat.reshape([1, _batchImgSize, _batchImgSize, 3])];
      } else {
        final float = Float32List(_batchImgSize * _batchImgSize * 3);
        int idx = 0;
        for (int y = 0; y < _batchImgSize; y++) {
          for (int x = 0; x < _batchImgSize; x++) {
            final p = resized.getPixel(x, y);
            float[idx++] = p.r / 255.0;
            float[idx++] = p.g / 255.0;
            float[idx++] = p.b / 255.0;
          }
        }
        inputTensor = [float.reshape([1, _batchImgSize, _batchImgSize, 3])];
      }

      List outputTensor;
      if (_batchIsInt8) {
        final out = Uint8List(1 * _batchClasses.length);
        outputTensor = [out.reshape([1, _batchClasses.length])];
      } else {
        final out = Float32List(1 * _batchClasses.length);
        outputTensor = [out.reshape([1, _batchClasses.length])];
      }

      _batchInterpreter!.run(inputTensor[0], outputTensor[0]);
      print('Batch: inference done');

      final rawScores = (outputTensor[0] as List).first as List;
      print('Batch: raw scores = $rawScores');

      int labelIdx = 0;
      double maxVal = -double.infinity;
      for (int i = 0; i < rawScores.length; i++) {
        final v = (rawScores[i] as num).toDouble();
        if (v > maxVal) { maxVal = v; labelIdx = i; }
      }

      double confidence;
      if (_batchIsInt8) {
        final outDetail = _batchInterpreter!.getOutputTensor(0);
        final scale     = outDetail.params.scale;
        final zeroPoint = outDetail.params.zeroPoint;
        if (scale > 0) {
          final floatScores = rawScores.map((v) =>
              ((v as num).toDouble() - zeroPoint) * scale).toList();
          final expScores = floatScores.map((v) => _expSafe(v)).toList();
          final sumExp    = expScores.fold<double>(0.0, (a, b) => a + b);
          confidence = sumExp > 0 ? expScores[labelIdx] / sumExp : 0.85;
        } else {
          confidence = (rawScores[labelIdx] as num).toDouble() / 255.0;
        }
      } else {
        final total = rawScores.fold<double>(0.0, (a, b) => a + (b as num).toDouble());
        confidence = total > 0 ? (rawScores[labelIdx] as num).toDouble() / total : 0.85;
      }

      if (labelIdx < 0 || labelIdx >= _batchClasses.length) {
        return {
          'isCloveDetected': false, 'grade': '', 'confidence': 0.0,
          'details': {}, 'error': AppStrings.current.mlErrorUnexpectedOutput, // ← was hardcoded
        };
      }

      final className = _batchClasses[labelIdx];
      print('Batch result: $className (index $labelIdx, confidence=${confidence.toStringAsFixed(3)})');

      if (className == 'Not_Clove') {
        return {
          'isCloveDetected': false,
          'grade': '',
          'confidence': 0.0,
          'details': {},
          'error': AppStrings.current.mlErrorNotClove,          // ← was hardcoded Swahili
        };
      }

      final mpetaRatio = (labelIdx == 3) ? 100.0 : 0.0;
      final gradingResult = _applyZSTCRules(mpetaRatio, className);

      final s = AppStrings.current;
      final auditTrail = s.batchAuditTrail(
        majorityGrade:   className,
        mpetaPercentage: mpetaRatio.toStringAsFixed(1),
        finalGrade:      gradingResult['grade'],
        reasoning:       gradingResult['reasoning'],
      );

      return {
        'isCloveDetected': true,
        'grade': gradingResult['grade'],
        'confidence': confidence,
        'details': {
          'color':   _analyzeColor(resized),
          'size':    _analyzeSize(resized),
          'defects': _detectDefects(resized),
        },
        'auditTrail': auditTrail,
        'error': null,
      };

    } catch (e, stack) {
      print('Batch error: $e');
      print('Stack: $stack');
      return {
        'isCloveDetected': false,
        'grade': '',
        'confidence': 0.0,
        'details': {},
        'error': AppStrings.current.mlErrorProcessing,          // ← was hardcoded Swahili
      };
    }
  }

  static double _expSafe(double x) {
    const double clip = 88.0;
    return x > clip ? clip : (x < -clip ? 0.0 : _dartExp(x));
  }

  static double _dartExp(double x) {
    return pow(2.718281828459045, x).toDouble();
  }

  // ── ZSTC rules — logic unchanged; reasoning strings now localised ─────────

  static Map<String, dynamic> _applyZSTCRules(
      double mpetaPercentage, String majorityGrade) {
    final s    = AppStrings.current;
    final val  = mpetaPercentage.toStringAsFixed(1);
    String grade, reasoning;
    double confidence;

    if (mpetaPercentage > gradeIIIMpetaMax) {
      grade      = 'Grade IV';
      confidence = 0.95;
      reasoning  = s.zstcReasonExceedsMpeta(val);              // ← was hardcoded Swahili
    } else if (mpetaPercentage > gradeIMpetaMax) {
      if (majorityGrade == 'Grade I' || majorityGrade == 'Grade II') {
        grade      = 'Grade II';
        confidence = 0.88;
        reasoning  = s.zstcReasonMidMpetaHigh(val, majorityGrade); // ← was hardcoded
      } else {
        grade      = 'Grade III';
        confidence = 0.85;
        reasoning  = s.zstcReasonMidMpetaHigh(val, majorityGrade);
      }
    } else {
      if (majorityGrade == 'Grade I') {
        grade      = 'Grade I';
        confidence = 0.95;
        reasoning  = s.zstcReasonGoodGradeI(val);              // ← was hardcoded Swahili
      } else if (majorityGrade == 'Grade II') {
        grade      = 'Grade II';
        confidence = 0.90;
        reasoning  = s.zstcReasonGoodGradeII(val);             // ← was hardcoded Swahili
      } else {
        grade      = 'Grade III';
        confidence = 0.85;
        reasoning  = s.zstcReasonGoodGradeLow(val);            // ← was hardcoded Swahili
      }
    }

    return {'grade': grade, 'confidence': confidence, 'reasoning': reasoning};
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  static Future<void> close() async {
    _interpreter?.close();
    _isInitialized = false;
    _batchInterpreter?.close();
    _batchInitialized = false;
  }
}

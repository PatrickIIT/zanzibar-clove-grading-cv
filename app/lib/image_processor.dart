// Updated ImageProcessor.dart
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

class ImageProcessor {
  
  static Future<File> optimizeImageForModel(File imageFile, {bool isBatchMode = false}) async {
    final tempDir = await getTemporaryDirectory();
    return await compute(_processInBackground, 
      _ProcessParams(imageFile, tempDir.path, isBatchMode: isBatchMode));
  }

  static Future<File> _processInBackground(_ProcessParams params) async {
    try {
      final bytes = await params.file.readAsBytes();
      final image = img.decodeImage(bytes);
      if (image == null) return params.file;

      print('Original image: ${image.width}x${image.height}');

      // Different target size based on mode
      final int targetSize = params.isBatchMode ? 640 : 480;   // 640 for batch is critical

      int targetWidth = image.width;
      int targetHeight = image.height;

      if (image.width > targetSize || image.height > targetSize) {
        if (image.width > image.height) {
          targetWidth = targetSize;
          targetHeight = (image.height * targetSize / image.width).round();
        } else {
          targetHeight = targetSize;
          targetWidth = (image.width * targetSize / image.height).round();
        }
      }

      final resized = img.copyResize(
        image, 
        width: targetWidth, 
        height: targetHeight,
        interpolation: img.Interpolation.average
      );
      
      print('Resized to: ${resized.width}x${resized.height}');

      final img.Image finalImage;
      if (params.isBatchMode) {
        finalImage = resized;   // Keep full image for batch
        print('Batch Mode: Keeping full rectangular image (no square crop)');
      } else {
        // Single mode - center crop to square
        final size = resized.width < resized.height ? resized.width : resized.height;
        final x = (resized.width - size) ~/ 2;
        final y = (resized.height - size) ~/ 2;
        
        finalImage = img.copyCrop(resized, x: x, y: y, width: size, height: size);
        print('Single Mode: Cropped to square ${finalImage.width}x${finalImage.height}');
      }

      final optimizedPath = '${params.tempDirPath}/opt_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final optimizedFile = File(optimizedPath);
      await optimizedFile.writeAsBytes(img.encodeJpg(finalImage, quality: 88));

      print('✅ Optimized: ${(await optimizedFile.length()) ~/ 1024}KB');
      return optimizedFile;

    } catch (e) {
      print('Image optimization error: $e');
      return params.file;
    }
  }

  // Keep these unchanged
  static img.Image resizeForModel(img.Image image) {
    return img.copyResize(image, width: 224, height: 224, interpolation: img.Interpolation.average);
  }

  static img.Image cropToSquare(img.Image image) {
    final size = image.width < image.height ? image.width : image.height;
    final x = (image.width - size) ~/ 2;
    final y = (image.height - size) ~/ 2;
    return img.copyCrop(image, x: x, y: y, width: size, height: size);
  }
}

class _ProcessParams {
  final File file;
  final String tempDirPath;
  final bool isBatchMode;

  _ProcessParams(this.file, this.tempDirPath, {this.isBatchMode = false});
}

import React, { useCallback, useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Upload, Image as ImageIcon, X, CheckCircle } from 'lucide-react';

interface ImageUploadProps {
  onImageUpload: (file: File) => void;
  uploadedImage: File | null;
  onRemoveImage: () => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageUpload,
  uploadedImage,
  onRemoveImage
}) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));

    if (imageFile) {
      onImageUpload(imageFile);
    }
  }, [onImageUpload]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageUpload(file);
    }
  }, [onImageUpload]);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  return (
    <Card className="diagnostic-card p-6">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-accent/20">
            <ImageIcon className="w-6 h-6 text-accent" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-foreground">Medical Imaging</h2>
            <p className="text-muted-foreground">Upload chest X-ray or CT scan for AI analysis</p>
          </div>
        </div>

        {!uploadedImage ? (
          /* Upload Area */
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`
              border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 cursor-pointer
              ${isDragOver
                ? 'border-primary bg-primary/5 glow-primary'
                : 'border-border/50 hover:border-primary/50 hover:bg-card/50'
              }
            `}
          >
            <div className="space-y-4">
              <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                <Upload className="w-8 h-8 text-primary" />
              </div>

              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-foreground">
                  Upload Medical Image
                </h3>
                <p className="text-muted-foreground">
                  Drag and drop your chest X-ray or CT scan here, or click to browse
                </p>
                <p className="text-sm text-muted-foreground">
                  Supports: JPG, PNG, DICOM (Max 10MB)
                </p>
              </div>

              <Button
                variant="outline"
                className="border-primary/50 text-primary hover:bg-primary/10"
                onClick={() => document.getElementById('file-input')?.click()}
              >
                Browse Files
              </Button>

              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                className="hidden"
              />
            </div>
          </div>
        ) : (
          /* Uploaded Image Preview */
          <div className="space-y-4">
            <div className="relative rounded-lg overflow-hidden bg-card border border-border/50">
              <img
                src={URL.createObjectURL(uploadedImage)}
                alt="Uploaded medical image"
                className="w-full h-64 object-contain bg-muted/20"
              />
              <Button
                variant="destructive"
                size="sm"
                className="absolute top-2 right-2"
                onClick={onRemoveImage}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            <div className="flex items-center justify-between p-4 bg-success/10 rounded-lg border border-success/20">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-success" />
                <div>
                  <p className="font-medium text-success-foreground">Image Uploaded Successfully</p>
                  <p className="text-sm text-muted-foreground">{uploadedImage.name}</p>
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => document.getElementById('file-input')?.click()}
                className="border-success/50 text-success hover:bg-success/10"
              >
                Replace Image
              </Button>
            </div>

            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
            />
          </div>
        )}
      </div>
    </Card>
  );
};

export default ImageUpload;

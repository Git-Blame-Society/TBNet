"use client";

import { X, Upload } from "lucide-react";

interface DataUploadProps {
  selectedSymptoms: string[];
  setSelectedSymptoms: (symptoms: string[]) => void;
  uploadedImage: File | null;
  setUploadedImage: (file: File | null) => void;
  onAnalyze: () => void;
}

export default function DataUpload({
  selectedSymptoms,
  setSelectedSymptoms,
  uploadedImage,
  setUploadedImage,
  onAnalyze,
}: DataUploadProps) {
  const symptomLabels = [
    "Fever for two weeks",
    "Coughing blood",
    "Sputum mixed with blood",
    "Night sweats",
    "Chest pain",
    "Back pain in certain parts",
    "Shortness of breath",
    "Weight loss",
    "Body feels tired",
    "Lumps that appear around the armpits and neck",
    "Cough and phlegm continuously for two weeks to four weeks",
    "Swollen lymph nodes",
    "Loss of appetite",
  ];

  const toggleSymptom = (label: string) => {
    if (selectedSymptoms.includes(label)) {
      setSelectedSymptoms(selectedSymptoms.filter((s) => s !== label));
    } else {
      setSelectedSymptoms([...selectedSymptoms, label]);
    }
  };

  return (
    <div className="border rounded-lg bg-white shadow-sm p-6 space-y-6">
      {/* Header with counter */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Symptom Assessment</h2>
        <span className="text-sm text-muted-foreground">{selectedSymptoms.length} selected</span>
      </div>

      {/* Symptom checkboxes */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {symptomLabels.map((label) => (
          <label
            key={label}
            className="flex items-center gap-3 px-3 py-2 border rounded-lg hover:bg-gray-50 cursor-pointer"
          >
            <input
              type="checkbox"
              checked={selectedSymptoms.includes(label)}
              onChange={() => toggleSymptom(label)}
              className="w-5 h-5 accent-black cursor-pointer"
            />
            <span className="text-sm">{label}</span>
          </label>
        ))}
      </div>

      {/* Upload section */}
      <div className="flex flex-col items-center">
        <h3 className="font-medium mb-2">Upload Chest X-ray</h3>

        {!uploadedImage ? (
          <label className="w-full flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-md p-6 cursor-pointer hover:border-gray-400">
            <Upload className="w-8 h-8 text-gray-500 mb-2" />
            <span className="text-sm text-gray-600">Click to Upload Image</span>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => setUploadedImage(e.target.files ? e.target.files[0] : null)}
            />
          </label>
        ) : (
          <div className="relative w-fit">
            <button
              onClick={() => setUploadedImage(null)}
              className="absolute -top-2 -right-2 bg-black text-white rounded-full p-1 hover:bg-gray-700"
            >
              <X className="w-4 h-4" />
            </button>

            <img
              src={URL.createObjectURL(uploadedImage)}
              alt="Uploaded preview"
              className="w-48 h-48 object-cover rounded-md border"
            />
          </div>
        )}
      </div>

      {/* Predict button */}
      <button
        onClick={onAnalyze}
        className="w-full px-4 py-2 bg-black text-white rounded hover:bg-gray-800"
      >
        Predict TB
      </button>
    </div>
  );
}

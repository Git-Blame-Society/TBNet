"use client";

import { FileText } from "lucide-react";

interface PredictionResultProps {
  result: {
    probability: number;
    verdict: "positive" | "negative" | "inconclusive";
    riskFactors: string[];
  } | null;
  isAnalyzing: boolean;
}

export default function PredictionResult({ result, isAnalyzing }: PredictionResultProps) {
  if (isAnalyzing) {
    return (
      <div className="border rounded-lg bg-white shadow-sm p-6 text-center text-gray-500">
        Analyzing data, please wait...
      </div>
    );
  }

  if (!result) {
    return (
      <div className="border rounded-lg bg-white shadow-sm p-6 text-center text-gray-500 flex flex-col items-center space-y-3">
        <FileText className="w-20 h-20 text-gray-400" />
        <h2 className="text-lg font-semibold text-gray-700">Awaiting Analysis</h2>
        <p>
          No prediction yet. Please upload data and click <b>Predict TB</b>.
        </p>
      </div>
    );
  }

  const probability = result.probability;
  const prediction = result.verdict === "positive" ? 1 : 0;
  const selectedSymptoms = result.riskFactors;

  const isHighRisk = probability > 50;

  return (
    <div className="border rounded-lg bg-white shadow-sm p-6 space-y-6">
      {/* Risk Header */}
      <div className="text-center">
        <h2
          className={`text-lg font-bold ${isHighRisk ? "text-red-600" : "text-green-600"}`}
        >
          {isHighRisk ? "High Risk of TB" : "Low Risk of TB"}
        </h2>
      </div>

      {/* Probability Progress */}
      <div>
        <h3 className="text-sm font-medium mb-2 text-gray-700">AI Probability</h3>
        <div className="w-full bg-gray-200 rounded-full h-5 overflow-hidden">
          <div
            className="h-5 flex items-center justify-center text-xs font-medium text-white bg-black"
            style={{ width: `${probability}%` }}
          >
            {probability}%
          </div>
        </div>
      </div>

      {/* Verdict Labels */}
      <div className="flex items-center justify-center gap-4">
        <span
          className={`px-4 py-1 rounded-full text-sm font-medium ${
            prediction === 0 ? "bg-black text-white" : "bg-gray-200 text-gray-600"
          }`}
        >
          Negative
        </span>
        <span
          className={`px-4 py-1 rounded-full text-sm font-medium ${
            prediction === 1 ? "bg-red-600 text-white" : "bg-gray-200 text-gray-600"
          }`}
        >
          Positive
        </span>
      </div>

      {/* Identified Factors */}
      <div>
        <h3 className="text-sm font-medium mb-2 text-gray-700">Identified Factors</h3>
        {selectedSymptoms.length > 0 ? (
          <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
            {selectedSymptoms.map((symptom, i) => (
              <li key={i}>{symptom}</li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-gray-400 italic">No symptoms selected.</p>
        )}
      </div>
    </div>
  );
}

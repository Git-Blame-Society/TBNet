'use client'
import { useState } from "react";

const Dashboard = () => {
  const [prob, setProb] = useState(0);
  const [pred, setPred] = useState(0);
  const [sym_prob, setSymProb] = useState(0);
  const [image_prob, setImageProb] = useState(0);
  const [image, setImage] = useState(null);

  const labels = [
    "fever_for_two_weeks",
    "coughing_blood",
    "sputum_mixed_with_blood",
    "night_sweats",
    "chest_pain",
    "back_pain_in_certain_parts",
    "shortness_of_breath",
    "weight_loss",
    "body_feels_tired",
    "lumps_that_appear_around_the_armpits_and_neck",
    "cough_and_phlegm_continuously_for_two_weeks_to_four_weeks",
    "swollen_lymph_nodes",
    "loss_of_appetite",
  ];

  // Store checkbox state in an array of booleans
  const [checked, setChecked] = useState(Array(labels.length).fill(false));

  const handleChange = (index) => {
    const newChecked = [...checked];
    newChecked[index] = !newChecked[index];
    setChecked(newChecked);
  };


const handleImageChange = (e) => {
  const file = e.target.files[0];
  console.log("Selected file:", file);
  setImage(file);
};

const handleDataUpload = async () => {
  const result = checked.map((val) => (val ? 1 : 0));
    console.log("Symptoms uploaded:", result);

    try {
      const res = await fetch("http://localhost:8000/upload-symptoms", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ result }),
      });

      if (res.ok) {
        let data = await res.json();
        console.log(data);
        setSymProb(data.sym_probability);
      }
    } catch (err) {
      console.log(err.message);
    }

  if (!image) {
    alert("Please select an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", image);

  try {
    const res = await fetch("http://localhost:8000/upload-image", {
      method: "POST",
      body: formData, // sending multipart/form-data
    });

    if (res.ok) {
      const data = await res.json();
      console.log("Image response:", data);
      console.log("progb==b", data.image_probability);
      setImageProb(data.image_probability);

      let final_prob = (sym_prob * 0.3) + (data.image_probability * 0.7);

      if(final_prob > 0.5){
        setPred(1);
      }
      else{
        setPred(0);
      }

      setProb(final_prob);
    }
  } catch (err) {
    console.error("Upload error:", err.message);
  }
};

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Select Symptoms</h2>
      <div className="grid gap-2">
        {labels.map((label, index) => (
          <label key={index} className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={checked[index]}
              onChange={() => handleChange(index)}
            />
            <span>{label.replaceAll("_", " ")}</span>
          </label>
        ))}
      </div>

      {/* Upload image section */}
<div className="mt-6">
  <h2 className="text-xl font-bold mb-2">Predict TB</h2>

  <input
    type="file"
    accept="image/*"
    id="fileInput"
    name="file"
    onChange={handleImageChange}
  />

  <button
    onClick={handleDataUpload}
    className="ml-2 px-4 py-2 bg-green-500 text-white rounded"
  >
    Predict TB
  </button>
</div>

      <p className="mt-4">Probability: {prob}</p>
      <p>Prediction: {pred}</p>
    </div>
  );
};

export default Dashboard;

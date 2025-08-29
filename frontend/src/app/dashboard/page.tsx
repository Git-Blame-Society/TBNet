'use client'
import { useState } from "react";

const Dashboard = () => {
  const [prob, setProb] = useState('');
  const [pred, setPred] = useState('');

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

  const handleUpload = async() => {
    const result = checked.map((val) => (val ? 1 : 0));
    console.log("Symptoms uploaded:", result);
    
    try{
const res = await fetch("http://localhost:8000/upload-symptoms", {
  method: "POST",
  headers: {
    Accept: "application/json",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ result }),
});

    if(res.ok){
      let data = await res.json();
      console.log(data);
      setProb(data.probability);
      setPred(data.prediction);
      }
    }
    catch(err){
      console.log(err.message);
    }

    return result;
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
      <button
        onClick={handleUpload}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
      >
        Upload Symptoms
      </button>
      <p>Probabilty: {prob}</p>
      <p>Prediction: {pred}</p>
    </div>
  );
};

export default Dashboard;

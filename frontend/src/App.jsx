import { useState } from 'react';
import Navbar from './Components/Navbar';
import Video from './Components/Video';
import './App.css';

function App() {
  const [keyframes, setKeyFrames] = useState([]);
  const [summary, setSummary] = useState("");

  return (
    <>
      <Navbar />
      <div className="windowsplit flex h-[94vh] bg-gray-100 ">
        <div className="columnsplit w-[40%]">
          <div className="videoinput border border-green-500 h-[55%]">
            <Video setSummary={setSummary} setKeyFrames={setKeyFrames} />
          </div>
          <div className="summaryarea border border-green-500 h-[45%] p-2">
            <h2>Summary</h2>
            <div className="summarytext bg-slate-300 rounded-lg p-2 mx-3 my-5 h-[80%]">
              {summary}
            </div>
          </div>
        </div>
        <div className="keyframes border border-green-500 w-[60%] p-2">
          <h2>Keyframes</h2>
          <div className="grid grid-cols-3 gap-4">
            {keyframes.map((keyframe, index) => (
              <img
                key={index}
                src={`data:image/jpeg;base64,${keyframe}`}
                alt={`Keyframe ${index + 1}`}
                className="w-full h-auto rounded shadow"
              />
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

export default App;

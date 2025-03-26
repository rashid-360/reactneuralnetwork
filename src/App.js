import React, { useRef, useEffect, useState } from 'react';
import jasonData from './mnist_weights_biases.json';
import * as math from 'mathjs';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale } from 'chart.js';
import './index.css';

// Register the necessary components for Chart.js
ChartJS.register(BarElement, CategoryScale, LinearScale);

const Canvas = (props) => {
  const canvasRef = useRef(null);
  const cwidth = 504;
  const cheight = 504;
  const rows = 28;
  const cubeSize = 18;
  const [r, setResult] = useState('');
  const [dd, setDd] = useState(null); // Initialize as null for conditional rendering

  const draw = (ctx, index) => {
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.fillRect((index % rows) * cubeSize, Math.floor(index / rows) * cubeSize, cubeSize, cubeSize);
    ctx.fill();
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const f = new Array(784).fill(0);

    context.canvas.width = cwidth;
    context.canvas.height = cheight;

    context.fillStyle = '#000000';
    context.fillRect(0, 0, context.canvas.width, context.canvas.height);

    const handleClick = (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      let i = Math.floor(x / cubeSize) + (Math.floor(y / cubeSize) * rows);
      f[i] = 1;

      draw(context, i);
    };

    canvas.addEventListener('mousedown', () => {
      canvas.addEventListener('mousemove', handleClick);
    });

    canvas.addEventListener('mouseup', () => {
      canvas.removeEventListener('mousemove', handleClick);
      test(f);
    });

    return () => {
      canvas.removeEventListener('mousemove', handleClick);
    };
  }, []);

  function test(input) {
    console.log('worked');
    const firstweightmultiplication = math.multiply(input, jasonData.dense_1.weights);
    const firstadd = math.add(firstweightmultiplication, jasonData.dense_1.biases);
    const firstfinal = firstadd.map(value => Math.max(0, value));
    const secondmultiplication = math.multiply(firstfinal, jasonData.dense_2.weights);
    const secondadd = math.add(secondmultiplication, jasonData.dense_2.biases);
    const result = secondadd.indexOf(Math.max(...secondadd));
    const res=secondadd.map((v)=>v+100)
    
    setResult(result);

    console.log(result, res);

    const ddd = {
      labels: res.map((_, index) => `${index}`), // Dynamic labels
      datasets: [
        {
          label: 'Predictions',
          data: secondadd,
          backgroundColor: 'rgba(75, 192, 192, 0.2)', 
          borderColor: 'rgba(75, 192, 192, 1)', 
          borderWidth: 1, 
        },
      ],
    };

    setDd(ddd); 
  }

  return (
    <div >
      <canvas className="canvas-container" ref={canvasRef} {...props} />
      <div className='con'>

      <div className='result'>
        <h1>Predicted Digit: {r}</h1>
      </div>
      {dd &&<div className='chart-container'> <Bar data={dd} /></div>} 
      </div>
    </div>
  );
};

export default Canvas;

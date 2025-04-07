import { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { ModelConfig } from '@/utils/model';

interface NetworkVisualizationProps {
  model: tf.Sequential | null;
  config: ModelConfig;
  inputFeatures: string[];
}

export default function NetworkVisualization({ model, config, inputFeatures }: NetworkVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  
  // 调整神经元布局参数，整体缩小
  const width = 500;
  const height = 300;
  const neuronRadius = 12;
  const layerSpacing = 120;
  const neuronSpacing = 40;
  
  useEffect(() => {
    if (!svgRef.current) return;
    
    // 获取权重（如果模型已训练）
    const weights = model?.layers.map(layer => {
      const w = layer.getWeights();
      return w.length > 0 ? w[0].arraySync() : null;
    }) || [];
    
    // 获取每层神经元数量
    const layerSizes = [2, ...config.hiddenLayers, 1];
    
    // 清空 SVG
    const svg = svgRef.current;
    svg.innerHTML = '';
    
    // 绘制每一层
    layerSizes.forEach((size, layerIndex) => {
      const layerX = 100 + layerIndex * layerSpacing;
      
      // 绘制该层的神经元
      for (let i = 0; i < size; i++) {
        const neuronY = height/2 + (i - (size-1)/2) * neuronSpacing;
        
        // 创建神经元圆圈
        const neuron = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        neuron.setAttribute("cx", layerX.toString());
        neuron.setAttribute("cy", neuronY.toString());
        neuron.setAttribute("r", neuronRadius.toString());
        neuron.setAttribute("fill", "#fff");
        neuron.setAttribute("stroke", "#666");
        neuron.setAttribute("stroke-width", "2");
        svg.appendChild(neuron);
        
        // 添加标签
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", layerX.toString());
        text.setAttribute("y", (neuronY + neuronRadius + 15).toString());
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("fill", "#666");
        text.setAttribute("font-size", "12");
        
        // 为输入层添加特征名称
        if (layerIndex === 0) {
          text.textContent = inputFeatures[i] || `输入 ${i+1}`;
        } else if (layerIndex === layerSizes.length - 1) {
          text.textContent = "通过概率";
        }
        svg.appendChild(text);
        
        // 绘制到下一层的连接
        if (layerIndex < layerSizes.length - 1) {
          const nextLayerSize = layerSizes[layerIndex + 1];
          const weightMatrix = weights[layerIndex];
          
          for (let j = 0; j < nextLayerSize; j++) {
            const nextNeuronY = height/2 + (j - (nextLayerSize-1)/2) * neuronSpacing;
            
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", (layerX + neuronRadius).toString());
            line.setAttribute("y1", neuronY.toString());
            line.setAttribute("x2", (layerX + layerSpacing - neuronRadius).toString());
            line.setAttribute("y2", nextNeuronY.toString());
            
            if (weightMatrix) {
              // 如果有权重，使用权重值设置线条样式
              const weight = weightMatrix[i] ? weightMatrix[i][j] : 0;
              const opacity = Math.min(Math.abs(weight), 1);
              line.setAttribute("stroke", weight > 0 ? "#4CAF50" : "#f44336");
              line.setAttribute("stroke-width", (Math.abs(weight) * 2).toString());
              line.setAttribute("stroke-opacity", opacity.toString());
            } else {
              // 如果没有权重，使用默认样式
              line.setAttribute("stroke", "#999");
              line.setAttribute("stroke-width", "1");
              line.setAttribute("stroke-opacity", "0.3");
            }
            
            svg.appendChild(line);
          }
        }
      }
      
      // 添加层标签
      const layerLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      layerLabel.setAttribute("x", layerX.toString());
      layerLabel.setAttribute("y", "30");
      layerLabel.setAttribute("text-anchor", "middle");
      layerLabel.setAttribute("fill", "#333");
      layerLabel.setAttribute("font-size", "14");
      layerLabel.setAttribute("font-weight", "bold");
      
      if (layerIndex === 0) {
        layerLabel.textContent = "输入层";
      } else if (layerIndex === layerSizes.length - 1) {
        layerLabel.textContent = "输出层";
      } else {
        layerLabel.textContent = `隐藏层 ${layerIndex}`;
      }
      svg.appendChild(layerLabel);
    });
  }, [model, config, inputFeatures]);

  return (
    <div className="bg-white p-4 rounded-lg shadow h-full">
      <h2 className="text-lg font-semibold mb-2">网络结构</h2>
      <div className="overflow-x-auto">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="mx-auto"
          style={{ minWidth: width }}
        />
      </div>
      <div className="mt-2 text-xs text-gray-600">
        <div className="flex items-center gap-2">
          <div className="flex items-center">
            <div className="w-3 h-1 bg-green-500 mr-1"></div>
            <span>正权重</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-1 bg-red-500 mr-1"></div>
            <span>负权重</span>
          </div>
          {!model && (
            <div className="flex items-center">
              <div className="w-3 h-1 bg-gray-400 mr-1"></div>
              <span>未训练</span>
            </div>
          )}
        </div>
        <p className="mt-1">
          注：线条的粗细表示权重的绝对值大小，颜色表示权重的正负
        </p>
      </div>
    </div>
  );
} 
import { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';

interface NetworkVisualizationProps {
  model: tf.LayersModel | null;
  config: {
    hiddenLayers: number[];
  };
  inputFeatures: string[];
}

interface LayerWeights {
  weights: number[][];
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
    const svg = svgRef.current;
    if (!svg) return;

    // 清除之前的内容
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }

    // 计算每一层的神经元数量
    const layerSizes = [2, ...config.hiddenLayers, 1];
    
    // 获取模型的权重
    const weights: LayerWeights[] = [];
    if (model) {
      model.layers.forEach((layer, i) => {
        if (i > 0) { // 跳过输入层
          const layerWeights = layer.getWeights();
          if (layerWeights.length > 0) {
            const w = layerWeights[0];
            weights.push({
              weights: Array.isArray(w) ? w : w.arraySync() as number[][]
            });
          }
        }
      });
    }

    // 绘制每一层
    layerSizes.forEach((size, layerIndex) => {
      const layerX = 60 + layerIndex * layerSpacing;
      const startY = (height - (size - 1) * neuronSpacing) / 2;

      // 绘制这一层的每个神经元
      for (let i = 0; i < size; i++) {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        const y = startY + i * neuronSpacing;
        
        circle.setAttribute("cx", layerX.toString());
        circle.setAttribute("cy", y.toString());
        circle.setAttribute("r", neuronRadius.toString());
        circle.setAttribute("fill", "#fff");
        circle.setAttribute("stroke", "#666");
        circle.setAttribute("stroke-width", "2");
        svg.appendChild(circle);

        // 添加标签
        if (layerIndex === 0) {
          // 输入层标签
          const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
          text.setAttribute("x", (layerX - neuronRadius - 5).toString());
          text.setAttribute("y", y.toString());
          text.setAttribute("text-anchor", "end");
          text.setAttribute("alignment-baseline", "middle");
          text.setAttribute("class", "text-sm");
          text.textContent = inputFeatures[i];
          svg.appendChild(text);
        } else if (layerIndex === layerSizes.length - 1) {
          // 输出层标签
          const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
          text.setAttribute("x", (layerX + neuronRadius + 5).toString());
          text.setAttribute("y", y.toString());
          text.setAttribute("text-anchor", "start");
          text.setAttribute("alignment-baseline", "middle");
          text.setAttribute("class", "text-sm");
          text.textContent = "通过概率";
          svg.appendChild(text);
        }

        // 如果不是最后一层，绘制到下一层的连接
        if (layerIndex < layerSizes.length - 1) {
          const nextLayerSize = layerSizes[layerIndex + 1];
          const nextLayerStartY = (height - (nextLayerSize - 1) * neuronSpacing) / 2;
          const weightMatrix = weights[layerIndex]?.weights;

          for (let j = 0; j < nextLayerSize; j++) {
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            const nextY = nextLayerStartY + j * neuronSpacing;
            
            line.setAttribute("x1", (layerX + neuronRadius).toString());
            line.setAttribute("y1", y.toString());
            line.setAttribute("x2", (layerX + layerSpacing - neuronRadius).toString());
            line.setAttribute("y2", nextY.toString());

            if (weightMatrix && Array.isArray(weightMatrix[i])) {
              // 如果有权重，使用权重值设置线条样式
              const weight = weightMatrix[i][j];
              const opacity = Math.min(Math.abs(weight), 1);
              line.setAttribute("stroke", weight > 0 ? "#4CAF50" : "#f44336");
              line.setAttribute("stroke-width", (Math.abs(weight) * 2).toString());
              line.setAttribute("stroke-opacity", opacity.toString());
            } else {
              // 默认样式
              line.setAttribute("stroke", "#ccc");
              line.setAttribute("stroke-width", "1");
            }
            
            svg.insertBefore(line, svg.firstChild); // 将线条放在最底层
          }
        }
      }
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
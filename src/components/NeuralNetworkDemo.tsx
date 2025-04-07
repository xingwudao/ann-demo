'use client';

import { useEffect, useState } from 'react';
import { loadData, prepareData, type DataPoint } from '@/utils/data';
import { createModel, trainModel, predict, type ModelConfig } from '@/utils/model';
import * as tf from '@tensorflow/tfjs';
import { Line } from 'react-chartjs-2';
import NetworkVisualization from './NetworkVisualization';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function NeuralNetworkDemo() {
  const [data, setData] = useState<DataPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [ranges, setRanges] = useState<any>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<Array<{epoch: number, loss: number, acc: number}>>([]);
  const [config, setConfig] = useState<ModelConfig>({
    hiddenLayers: [8, 4],
    learningRate: 0.01
  });
  const [inputValues, setInputValues] = useState({
    hoursStudied: 20,
    attendanceRate: 80
  });

  useEffect(() => {
    const initializeData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const loadedData = await loadData();
        console.log('数据加载成功:', loadedData.length, '条记录');
        setData(loadedData);
      } catch (err) {
        console.error('数据加载失败:', err);
        setError(err instanceof Error ? err.message : '数据加载失败');
      } finally {
        setIsLoading(false);
      }
    };

    // 设置 TensorFlow.js 后端并抑制日志
    tf.setBackend('webgl').then(() => {
      tf.env().set('DEBUG', false);
      tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
      initializeData();
    });
  }, []);

  const handleTrain = async () => {
    if (isLoading) {
      console.error('数据正在加载中');
      return;
    }
    
    if (!data.length) {
      console.error('没有训练数据');
      return;
    }

    try {
      // 开始新的训练时，清空历史记录
      setTrainingHistory([]);
      
      console.log('开始准备训练数据...');
      const { xsTensor, ysTensor, ranges: dataRanges } = prepareData(data);
      console.log('训练数据范围:', dataRanges);
      
      setRanges(dataRanges);

      console.log('创建模型...');
      const newModel = createModel(config);
      setModel(newModel);

      console.log('开始训练...');
      await trainModel(newModel, xsTensor, ysTensor, 50, (epoch, logs) => {
        if (logs) {
          setTrainingHistory(prev => [...prev, {
            epoch,
            loss: logs.loss || 0,
            acc: logs.acc || logs.accuracy || 0
          }]);
        }
      });

      console.log('训练完成');
      
      // 清理张量
      xsTensor.dispose();
      ysTensor.dispose();
      
    } catch (error) {
      console.error('训练过程出错:', error);
      setModel(null);
      setRanges(null);
    }
  };

  const handlePredict = () => {
    if (!model || !ranges) {
      console.error('模型未训练');
      return;
    }

    try {
      console.log('开始预测...');
      console.log('使用的数据范围:', ranges);
      
      const result = predict(model, [
        inputValues.hoursStudied,
        inputValues.attendanceRate
      ], ranges);

      if (isNaN(result)) {
        console.error('预测结果无效');
        setPrediction(0);
      } else {
        console.log('预测结果:', result);
        setPrediction(result);
      }
    } catch (error) {
      console.error('预测过程出错:', error);
      setPrediction(0);
    }
  };

  return (
    <div className="w-full space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">模型配置</h2>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  第一隐藏层神经元数量
                </label>
                <input
                  type="number"
                  value={config.hiddenLayers[0]}
                  onChange={(e) => {
                    const value = Math.min(8, Math.max(1, Number(e.target.value)));
                    setConfig({
                      ...config,
                      hiddenLayers: [
                        value,
                        config.hiddenLayers[1]
                      ]
                    });
                  }}
                  min="1"
                  max="8"
                  step="1"
                  className="w-full p-2 border rounded"
                />
                <p className="text-xs text-gray-500 mt-1">最大值: 8</p>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">
                  第二隐藏层神经元数量
                </label>
                <input
                  type="number"
                  value={config.hiddenLayers[1]}
                  onChange={(e) => {
                    const value = Math.min(8, Math.max(1, Number(e.target.value)));
                    setConfig({
                      ...config,
                      hiddenLayers: [
                        config.hiddenLayers[0],
                        value
                      ]
                    });
                  }}
                  min="1"
                  max="8"
                  step="1"
                  className="w-full p-2 border rounded"
                />
                <p className="text-xs text-gray-500 mt-1">最大值: 8</p>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                学习率
              </label>
              <input
                type="number"
                value={config.learningRate}
                onChange={(e) => setConfig({
                  ...config,
                  learningRate: Number(e.target.value)
                })}
                step="0.001"
                className="w-full p-2 border rounded"
              />
            </div>
            <button
              onClick={handleTrain}
              disabled={isLoading}
              className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isLoading ? '数据加载中...' : '训练模型'}
            </button>
            {error && (
              <div className="text-red-500 text-sm mt-2">
                {error}
              </div>
            )}
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">预测</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                每周学习时间（小时）
              </label>
              <input
                type="number"
                value={inputValues.hoursStudied}
                onChange={(e) => setInputValues({
                  ...inputValues,
                  hoursStudied: Number(e.target.value)
                })}
                className="w-full p-2 border rounded"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                出勤率（%）
              </label>
              <input
                type="number"
                value={inputValues.attendanceRate}
                onChange={(e) => setInputValues({
                  ...inputValues,
                  attendanceRate: Number(e.target.value)
                })}
                className="w-full p-2 border rounded"
              />
            </div>
            <button
              onClick={handlePredict}
              disabled={!model}
              className="w-full bg-green-500 text-white py-2 rounded hover:bg-green-600 disabled:bg-gray-400"
            >
              预测结果
            </button>
            {prediction !== null && (
              <div className="mt-4 p-4 bg-gray-100 rounded">
                <p className="text-center">
                  通过概率：{(prediction * 100).toFixed(2)}%
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 网络可视化和训练过程并排显示 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <NetworkVisualization
          model={model}
          config={config}
          inputFeatures={['学习时间', '出勤率']}
        />

        {trainingHistory.length > 0 && (
          <div className="bg-white p-4 rounded-lg shadow h-full">
            <h2 className="text-lg font-semibold mb-2">训练过程</h2>
            <div style={{ position: 'relative', height: '250px' }}>
              <Line
                data={{
                  labels: trainingHistory.map(h => h.epoch),
                  datasets: [
                    {
                      label: '损失值',
                      data: trainingHistory.map(h => h.loss),
                      borderColor: 'rgb(255, 99, 132)',
                      tension: 0.1,
                      yAxisID: 'y-loss'
                    },
                    {
                      label: '准确率',
                      data: trainingHistory.map(h => h.acc),
                      borderColor: 'rgb(75, 192, 192)',
                      tension: 0.1,
                      yAxisID: 'y-accuracy'
                    }
                  ]
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  animation: false,
                  elements: {
                    point: {
                      radius: 0
                    }
                  },
                  plugins: {
                    legend: {
                      position: 'top',
                      align: 'center'
                    }
                  },
                  scales: {
                    'y-loss': {
                      type: 'linear',
                      position: 'left',
                      min: 0,
                      max: 2,
                      title: {
                        display: true,
                        text: '损失值'
                      },
                      ticks: {
                        maxTicksLimit: 5
                      }
                    },
                    'y-accuracy': {
                      type: 'linear',
                      position: 'right',
                      min: 0,
                      max: 1,
                      title: {
                        display: true,
                        text: '准确率'
                      },
                      grid: {
                        drawOnChartArea: false
                      },
                      ticks: {
                        maxTicksLimit: 5
                      }
                    },
                    x: {
                      ticks: {
                        maxTicksLimit: 10
                      }
                    }
                  }
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 
import * as tf from '@tensorflow/tfjs';

export interface ModelConfig {
  hiddenLayers: number[];
  learningRate: number;
}

export function createModel(config: ModelConfig) {
  const model = tf.sequential();
  
  // 输入层
  model.add(tf.layers.dense({
    inputShape: [2],
    units: config.hiddenLayers[0],
    activation: 'relu',
  }));
  
  // 隐藏层
  for (let i = 1; i < config.hiddenLayers.length; i++) {
    model.add(tf.layers.dense({
      units: config.hiddenLayers[i],
      activation: 'relu',
    }));
  }
  
  // 输出层
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
  }));
  
  const optimizer = tf.train.adam(config.learningRate);
  
  model.compile({
    optimizer: optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  
  return model;
}

export async function trainModel(
  model: tf.Sequential,
  xsTrain: tf.Tensor2D,
  ysTrain: tf.Tensor2D,
  epochs: number,
  onEpochEnd?: (epoch: number, logs: tf.Logs) => void
) {
  await model.fit(xsTrain, ysTrain, {
    epochs,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpochEnd && logs) {
          onEpochEnd(epoch, logs);
        }
      },
    },
  });
}

export function predict(
  model: tf.Sequential,
  input: number[],
  ranges: {
    hours: { min: number; max: number };
    attendance: { min: number; max: number };
  }
) {
  // 检查输入值是否在合理范围内
  if (input[0] < 0 || input[0] > 40 || input[1] < 0 || input[1] > 100) {
    console.error('输入值超出合理范围');
    return 0;
  }

  console.log('预测输入值:', input);
  console.log('数据范围:', ranges);

  // 标准化输入数据
  const normalizedInput = [
    (input[0] - ranges.hours.min) / (ranges.hours.max - ranges.hours.min),
    (input[1] - ranges.attendance.min) / (ranges.attendance.max - ranges.attendance.min),
  ];

  console.log('标准化后的输入值:', normalizedInput);
  
  // 检查标准化后的值
  if (normalizedInput.some(value => !isFinite(value))) {
    console.error('数据标准化出错，可能是范围值相等导致除以零');
    return 0;
  }

  const inputTensor = tf.tensor2d([normalizedInput]);
  const prediction = model.predict(inputTensor) as tf.Tensor;
  const result = prediction.dataSync()[0];
  
  // 清理张量
  inputTensor.dispose();
  prediction.dispose();
  
  // 确保结果在 0-1 之间
  return Math.max(0, Math.min(1, result));
} 
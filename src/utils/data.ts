import * as tf from '@tensorflow/tfjs';
import Papa from 'papaparse';

export interface DataPoint {
  hoursStudied: number;
  attendanceRate: number;
  passed: number;
}

export async function loadData(): Promise<DataPoint[]> {
  try {
    const response = await fetch('/data.csv');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const csv = await response.text();
    
    const results = Papa.parse(csv, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true
    });

    console.log('解析的CSV数据:', results.data);

    const mappedData = results.data
      .filter((row: any) => 
        row.hours_studied_per_week != null && 
        row.attendance_rate_percent != null && 
        row.passed_exam != null
      )
      .map((row: any) => ({
        hoursStudied: Number(row.hours_studied_per_week),
        attendanceRate: Number(row.attendance_rate_percent),
        passed: Number(row.passed_exam)
      }));

    console.log('处理后的数据:', mappedData);
    
    if (mappedData.length === 0) {
      throw new Error('没有有效的训练数据');
    }

    return mappedData;
  } catch (error) {
    console.error('加载数据时出错:', error);
    throw error;
  }
}

export function normalizeData(data: DataPoint[]) {
  if (!data || data.length === 0) {
    throw new Error('没有训练数据');
  }

  const hoursStudied = data.map(d => d.hoursStudied);
  const attendanceRate = data.map(d => d.attendanceRate);

  // 检查是否有无效值
  if (hoursStudied.some(h => isNaN(h) || h === null)) {
    throw new Error('存在无效的学习时间数据');
  }
  if (attendanceRate.some(a => isNaN(a) || a === null)) {
    throw new Error('存在无效的出勤率数据');
  }

  const minHours = Math.min(...hoursStudied);
  const maxHours = Math.max(...hoursStudied);
  const minAttendance = Math.min(...attendanceRate);
  const maxAttendance = Math.max(...attendanceRate);

  // 检查数据范围的有效性
  if (!isFinite(minHours) || !isFinite(maxHours) || maxHours === minHours) {
    throw new Error('学习时间数据范围无效');
  }
  if (!isFinite(minAttendance) || !isFinite(maxAttendance) || maxAttendance === minAttendance) {
    throw new Error('出勤率数据范围无效');
  }

  console.log('数据范围:', {
    hours: { min: minHours, max: maxHours },
    attendance: { min: minAttendance, max: maxAttendance }
  });

  return {
    normalizedData: data.map(d => ({
      hoursStudied: (d.hoursStudied - minHours) / (maxHours - minHours),
      attendanceRate: (d.attendanceRate - minAttendance) / (maxAttendance - minAttendance),
      passed: d.passed,
    })),
    ranges: {
      hours: { min: minHours, max: maxHours },
      attendance: { min: minAttendance, max: maxAttendance },
    },
  };
}

export function prepareData(data: DataPoint[]) {
  const { normalizedData, ranges } = normalizeData(data);
  
  const xsTensor = tf.tensor2d(
    normalizedData.map(d => [d.hoursStudied, d.attendanceRate])
  );
  
  const ysTensor = tf.tensor2d(
    normalizedData.map(d => [d.passed]),
    [normalizedData.length, 1]
  );

  return {
    xsTensor,
    ysTensor,
    ranges,
  };
} 
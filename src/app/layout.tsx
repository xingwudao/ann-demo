import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "神经网络演示",
  description: "使用神经网络预测考试通过概率",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh">
      <body>{children}</body>
    </html>
  );
}

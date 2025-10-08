#! /ust/bin/env pyhton3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pyod.models.ecod import ECOD


def _create_folder(out):
    """指定されたフォルダが無ければ作成し、フォルダの絶対パスを返す関数"""
    abs_path = os.path.abspath(out)
    if os.path.exists(abs_path):
        print(f"* output folder {abs_path} is already exists.")
    else:
        print(f"* create new output folder {abs_path}.")
        os.makedirs(abs_path,exist_ok=False)

    return abs_path

def train_model(args):
    print("* training mode started.")

    # 学習用データの読み込み
    data_path = os.path.abspath(args.data)
    df = pd.read_csv(data_path)
    print(f"* taining data loaded from {data_path}")

    # 学習の実行
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    clf = ECOD()
    clf.fit(X_scaled)
    print("* training done.")

    # 学習データへの推論結果の保存
    out_path = _create_folder(args.out)
    result_path = os.path.join(out_path,"result.csv")
    result = pd.Series(clf.predict(df), index=df.index, name="class")
    result.to_csv(result_path,index=None)
    print(f"* result save to {out_path}")

    # 学習済みモデルの保存
    model_path = os.path.abspath(args.model)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"* model saved to {model_path}")

    return

def eval_model(args):
    print("* evaluation mode started.")

    # 評価用データの読み込み
    data_path = os.path.abspath(args.data)
    df = pd.read_csv(data_path)
    print(f"* evaluation data loaded from {data_path}")

    # 学習済みモデルの読み込み
    model_path = os.path.abspath(args.model)
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    print(f"* trained model loaded from {model_path}")

    # 評価の実行
    df["class"] = clf.predict(df)
    print("* evaluation done.")
    
    # 評価結果の保存
    out_path = _create_folder(args.out)
    png_path = os.path.join(out_path, "histogram.png")
    plt.figure()
    sns.histplot(clf.decision_scores_, bins = 50, kde=True)
    plt.vlines(clf.threshold_, ymin=0, ymax=90, color='red')
    plt.savefig(png_path)
    plt.close()
    # ここを実装する
    print(f"* result save to {out_path}")

    return

def infer_model(args):
    print("* inference mode started.")

    # 推論用データの読み込み
    data_path = os.path.abspath(args.data)
    df = pd.read_csv(data_path)
    print(f"* inference data loaded from {data_path}")

    # 学習済みモデルの読み込み
    model_path = os.path.abspath(args.model)
    # ここを実装する
    print(f"* trained model loaded from {model_path}")

    # 推論の実行
    out_path = _create_folder(args.out)
    result_path = os.path.join(out_path, "result.csv")
    result = pd.Series(clf.predict(df), index=df.index, name="class")
    result.to_csv(result_path,index=None)
    
    print("* inference done.")

    # 推論結果の保存
    out_path = _create_folder(args.out)
    png_path = os.path.join(out_path, "scatterplot.png")
    # ここを実装する
    print(f"* result save to {out_path}")

    return

if __name__ == "__main__":
    """コマンドラインでの実行"""

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="異常検知モデルの学習、評価、推論を行う")
    parser.add_argument("mode", choices=["train", "eval", "infer"],
                        help="学習、評価、推論のうちからどの機能を呼び出すかを指定\ntrain : 学習\neval : 評価\ninfer : 推論")
    parser.add_argument("data", 
                        help="学習、評価、推論を行うデータのファイルパス")
    parser.add_argument("model", 
                        help="学習済みモデルのファイルパス\n学習モードでは新規作成される\n評価、推論モードでは読み込みを行う")
    parser.add_argument("-o","--out", default="out", 
                        help="評価、推論モードの出力ファイル群を保存するフォルダのパス\nデフォルトはカレントディレクトリに`out`というフォルダが新規作成される")
    # コマンドライン引数の読み込み
    args = parser.parse_args()
    # 各モードでの実行
    if args.mode == "train":
        train_model(args)
    elif args.mode == "eval":
        eval_model(args)
    elif args.mode == "infer":
        infer_model(args)
    else:
        pass
    print("All processes completed.")
        

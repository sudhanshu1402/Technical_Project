# 🧠 Technical Project: SPAAC 
~ A smart EYE for every Class

**SPAAC** is an advanced technical project focusing on computer vision and potential emotion recognition applications. It involves deep learning models (`.hdf5`), data augmentation, and visual processing utilities.

## 📂 Project Structure

| Directory/File | Description |
| :--- | :--- |
| **models/** | Pre-trained deep learning models (e.g., `emotion_model.hdf5`). |
| **utils/** | Helper scripts for inference, visualization, and preprocessing. |
| **images/** | Dataset or sample images used for testing/training. |
| **test/** | Test assets including videos (`.mp4`) and GIFs. |
| **SPAAC_App.py** | Main application entry point. |

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** OpenCV (`cv2`)
- **Language:** Python
- **Visualization:** Matplotlib / Custom Visualizers

<img width="5972" height="2428" alt="image" src="https://github.com/user-attachments/assets/9951a1fd-4407-478b-98f5-3fe59756b3cd" />

## 🚀 Overview

This project implements a pipeline for analyzing visual data. It includes modules for:
- Data Augmentation
- Gradient Class Activation Mapping (Grad-CAM)
- Real-time Inference on video streams

In today’s time, there are several students who don’t pay attention during class lectures. Teacher can’t pay attention to each student specifically in the class. So, to overcome this gap and to fill it we came up with the idea of this project - “Student Performance Analysis and Attention in Classroom (SPAAC).

SPAAC is an idea to make a real time system which can calculate student attendance along with its performance (i.e., happy, neutral, sad or angry) in the classroom with the help of classroom CCTV using facial recognition algorithms along with deep learning to achieve better results.

This system will capture students through a CCTV camera present in the classroom and generate emotions based on their expression in class. Algorithms like CNN, Image Classification, Score Prediction, Face Recognition, Regression, SVM, Naive Bayes, Accuracy and Prediction are used. During the first phase we have studied various aspects and implemented a very basic facial recognition program to recognize the individual.

Face systems with artificial intelligence are dramatically changing businesses. There is a wide range of face system building platforms that are available for various enterprises, such as e-commerce, retail, banking, leisure, travel, healthcare, and so on.

---
*Maintained by Sudhanshu Singh*


import { Response } from 'express';
import mongoose from 'mongoose';
import { User, Customer, Transaction } from '../models';
import { AuthRequest, Role } from '../types';

export const addCustomer = async (req: AuthRequest, res: Response) => {
  const { name } = req.body;
  const retailerId = req.user?.id;

  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    // 1 Point = 1 Customer
    const retailer = await User.findOneAndUpdate(
      { _id: retailerId, points: { $gte: 1 } },
      { $inc: { points: -1 } },
      { session, new: true }
    );

    if (!retailer) throw new Error("Insufficient points");

    const customer = await Customer.create([{ name, retailerId }], { session });

    await session.commitTransaction();
    res.status(201).json(customer[0]);
  } catch (error: any) {
    await session.abortTransaction();
    res.status(400).json({ error: error.message });
  } finally {
    session.endSession();
  }
};

export const managePoints = async (req: AuthRequest, res: Response) => {
  const { retailerId, amount, type } = req.body; // type: 'TRANSFER' | 'REVERT'
  
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const incAmount = type === 'TRANSFER' ? amount : -amount;

    const updatedUser = await User.findOneAndUpdate(
      { _id: retailerId, points: { $gte: type === 'REVERT' ? amount : 0 } },
      { $inc: { points: incAmount } },
      { session, new: true }
    );

    if (!updatedUser) throw new Error("Operation failed: Check retailer balance");

    await Transaction.create([{
      type, amount, adminId: req.user?.id, retailerId
    }], { session });

    await session.commitTransaction();
    res.json({ message: "Success", balance: updatedUser.points });
  } catch (error: any) {
    await session.abortTransaction();
    res.status(400).json({ error: error.message });
  } finally {
    session.endSession();
  }
};


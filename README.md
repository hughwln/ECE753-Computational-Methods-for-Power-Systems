# Homework setup for ECE753 at North Carolina State University.
### /data

Yearly 15-min load profile, file days of them have missing values from 10:00 to 13:45. Missing data is set to zero.

Ground truth will be uploaded after the due date.

### /matlab_code
test.m  the script used to grade the homework.

example.m  a simple example code to restore missing values. You can refer to it or write your own code, totally up to you.

### /BERT-PIN(Python)
**This part is not mandatory. It's another example in python.**

If you use this code, please cite this paper:
_Yi Hu, Kai Ye, Hyeonjin Kim, and Ning Lu, “BERT-PIN: A BERT-based Framework for Recovering Missing Data Segments in Time-series Load Profiles”, available at http://arxiv.org/abs/2310.17742_

This is a simplified version of BERT-PIN, for full version (this homework doesn't need it) you can refer to https://github.com/hughwln/BERT-PIN_public.

It's a well-trained model, you can run _bert-pin.py_ directly and get a good result.

Note that you need a python environment to run the code.

There are some tips for configuring a python environment (you can google for how to perform each step, if you need):
1. Install Anaconda. It's a perfect tool to manage your python environment.
2. Open Anaconda, and create a virtual environment. I recommend you use the second latest python version.
3. Switch to your virtual environment and install the packages: pytorch, matplotlib, pandas, numpy. (you can install some others if you need).
4. Run the code using IDE, such as Pycharm, Spyder, Sublime text, and so on. Or you can run the code in Anaconda Prompt.

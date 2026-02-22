# Baum Welch Algorithm Visualizer

**Name :** Nandana Sasikumar

**University Registration Number :** TCR24CS050

**Course :** Pattern Recognition

**Institution :** Government Engineering College, Thrissur

---

### Project Overview
This project implements the Baum-Welch algorithm, an Expectation-Maximization (EM) technique used to learn the parameters of a Hidden Markov Model (HMM) when only observation sequences are available. The implementation is a web-based application built using Streamlit, allowing for interactive parameter tuning and real-time visualization of model convergence.


### Technical Features
The application performs the following core HMM calculations as defined in the course material:
* **Forward Probability ($\alpha$):** Measures the probability of the prefix evidence.
* **Backward Probability ($\beta$):** Measures the probability of the suffix evidence.
* **Soft Credit Assignment ($\gamma$):** Calculates state responsibility at each time step.
* **Parameter Updates:** Iteratively updates the initial probabilities ($\pi$), transition matrix ($A$), and emission matrix ($B$).


### Installation & Execution
**Prerequisites**

Ensure you have Python installed. You will need the following libraries:
* streamlit
* numpy
* matplotlib
* pandas

Follow these steps to set up and run the application on your local machine:

**1. Prepare the Project Directory:**
  * Create a dedicated folder for the project on your computer.
  * Ensure the provided Python script is saved inside this folder and named exactly `app.py`.
    
**2. Install Dependencies:**
  Open your terminal or command prompt and run the following command to install the necessary Python libraries:

  ```bash
  pip install streamlit numpy matplotlib pandas
  ```

**3. Run the Application:**
  * Navigate the terminal to the project directory where your `app.py` file is saved.
  * Execute the following command to launch the web interface:

  ```bash
  streamlit run app.py
  ```
  * The application will automatically open in your default web browser. If it does not, copy the "Local URL" provided in the terminal and paste it into your browser's address bar.

    

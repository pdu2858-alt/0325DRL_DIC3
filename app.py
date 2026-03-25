import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定 Streamlit 頁面
st.set_page_config(page_title="Bandit Strategies Comparison", layout="wide")

st.title("🎰 A/B Testing vs. Multi-Armed Bandits (6 Strategies)")
st.markdown("""
### 🧩 Problem Setup
* **Total budget:** 10,000 (10,000 rounds)
* **Bandits True Expected Returns:** Arm A: **0.8**, Arm B: **0.7**, Arm C: **0.5**
* **A/B Test Rule:** First 2,000 budget split equally between A and B (Ignore C). Remaining 8,000 goes to the winner.
""")

st.divider()

# ==========================================
# 任務 1~5：分析與計算區塊 (Analytical Solution)
# ==========================================
st.header("📝 Analytical Solution (Tasks 1 to 5)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Task 1: A/B Test Phase (Exploration)")
    st.markdown("""
    We allocate 1,000 to Arm A and 1,000 to Arm B.
    * Expected Reward from A = 1000 * 0.8 = 800
    * Expected Reward from B = 1000 * 0.7 = 700
    * **Total Expected Exploration Reward = 1,500**
    """)
    
    st.subheader("Task 2: Bandit Selection")
    st.markdown("""
    Since the true mean of Arm A (0.8) is greater than Arm B (0.7), over 1,000 trials, the Law of Large Numbers dictates that the empirical mean of A will very likely be higher. 
    * **Selected Bandit for Exploitation: Arm A**
    """)
    
    st.subheader("Task 3: Exploitation Phase & Total Reward")
    st.markdown("""
    The remaining 8,000 is allocated entirely to Arm A.
    * Expected Exploitation Reward = 8000 * 0.8 = 6400
    * **Total Expected Reward = 1500 (Exploration) + 6400 (Exploitation) = 7,900**
    """)

with col2:
    st.subheader("Task 4: Optimal Strategy Comparison")
    st.markdown("""
    The optimal strategy is to know the best arm from the start and allocate all 10,000 to Arm A.
    * **Optimal Expected Reward = 10000 * 0.8 = 8,000**
    """)
    
    st.subheader("Task 5: Regret of A/B Testing")
    st.markdown("""
    Regret is the difference between the Optimal Reward and the Total Expected Reward of our strategy.
    * **Regret** = 8000 - 7900 = **100**
    *(Note: The 100 regret comes entirely from pulling Arm B 1,000 times during exploration, losing 0.1 per pull).*
    """)

st.divider()

# ==========================================
# 任務 6：解釋與圖表證明區塊 (Explanation & Visual Proof)
# ==========================================
st.header("🧠 Task 6: How MAB Outperforms A/B Testing")
st.markdown("""
**A/B testing is static.** It rigidly forces you to spend 1,000 on Arm B even if it becomes obvious early on that Arm A is better. 

**Bandit Algorithms (ε-Greedy, UCB, Thompson Sampling, etc.) are adaptive:**
* **Dynamic Resource Allocation:** They update their confidence after *every single pull*. 
* **Minimizing Waste:** As soon as Arm A starts looking like the winner, they shift more budget to Arm A *during* the exploration phase.
* **Result:** They don't waste 1,000 full pulls on a suboptimal arm, resulting in a **lower cumulative regret**.
""")

# ==========================================
# 模擬演算法 (Simulation Functions) - 包含 6 種策略
# ==========================================
BUDGET = 10000
TRUE_MEANS = [0.8, 0.7, 0.5]
OPTIMAL_MEAN = max(TRUE_MEANS)

def simulate_ab_test():
    rewards = np.zeros(BUDGET)
    pulls_A = np.random.binomial(1, TRUE_MEANS[0], 1000)
    pulls_B = np.random.binomial(1, TRUE_MEANS[1], 1000)
    rewards[0:1000] = pulls_A
    rewards[1000:2000] = pulls_B
    mean_A = np.mean(pulls_A)
    mean_B = np.mean(pulls_B)
    best_arm = 0 if mean_A >= mean_B else 1
    rewards[2000:] = np.random.binomial(1, TRUE_MEANS[best_arm], BUDGET - 2000)
    return rewards

def simulate_optimistic_initial_values():
    Q = np.array([5.0, 5.0, 5.0]) # 樂觀的初始值
    N = np.zeros(3)
    rewards = np.zeros(BUDGET)
    for t in range(BUDGET):
        action = np.argmax(Q)
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        N[action] += 1
        Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])
    return rewards

def simulate_epsilon_greedy(epsilon=0.1):
    Q = np.zeros(3)
    N = np.zeros(3)
    rewards = np.zeros(BUDGET)
    for t in range(BUDGET):
        if np.random.rand() < epsilon:
            action = np.random.choice(3)
        else:
            action = np.argmax(Q)
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        N[action] += 1
        Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])
    return rewards

def simulate_softmax(tau=0.1):
    Q = np.zeros(3)
    N = np.zeros(3)
    rewards = np.zeros(BUDGET)
    for t in range(BUDGET):
        # 避免溢位 (Prevent overflow)
        exp_Q = np.exp((Q - np.max(Q)) / tau) 
        probs = exp_Q / np.sum(exp_Q)
        action = np.random.choice(3, p=probs)
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        N[action] += 1
        Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])
    return rewards

def simulate_ucb(c=2.0):
    Q = np.zeros(3)
    N = np.zeros(3)
    rewards = np.zeros(BUDGET)
    for t in range(3):
        action = t
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        N[action] += 1
        Q[action] = reward
    for t in range(3, BUDGET):
        ucb_values = Q + c * np.sqrt(np.log(t) / N)
        action = np.argmax(ucb_values)
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        N[action] += 1
        Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])
    return rewards

def simulate_thompson_sampling():
    alpha = np.ones(3)
    beta = np.ones(3)
    rewards = np.zeros(BUDGET)
    for t in range(BUDGET):
        sampled_theta = np.random.beta(alpha, beta)
        action = np.argmax(sampled_theta)
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        if reward == 1:
            alpha[action] += 1
        else:
            beta[action] += 1
    return rewards

# ==========================================
# 執行模擬與繪圖 (Visual Proof)
# ==========================================
st.sidebar.header("Simulation Settings")
runs = st.sidebar.slider("Number of Simulations to Average", min_value=1, max_value=50, value=10)

if st.button("Run Simulation (6 Strategies)"):
    with st.spinner('Simulating 10,000 rounds for all 6 algorithms...'):
        results = {
            "A/B Testing": np.zeros(BUDGET),
            "Optimistic Init": np.zeros(BUDGET),
            "ε-Greedy (ε=0.1)": np.zeros(BUDGET),
            "Softmax (τ=0.1)": np.zeros(BUDGET),
            "UCB (c=2)": np.zeros(BUDGET),
            "Thompson Sampling": np.zeros(BUDGET)
        }
        
        for _ in range(runs):
            results["A/B Testing"] += simulate_ab_test()
            results["Optimistic Init"] += simulate_optimistic_initial_values()
            results["ε-Greedy (ε=0.1)"] += simulate_epsilon_greedy()
            results["Softmax (τ=0.1)"] += simulate_softmax()
            results["UCB (c=2)"] += simulate_ucb()
            results["Thompson Sampling"] += simulate_thompson_sampling()
            
        for key in results:
            results[key] /= runs

        optimal_cumulative_reward = np.cumsum(np.full(BUDGET, OPTIMAL_MEAN))
        
        regret_results = {}
        for key in results:
            cumulative_reward = np.cumsum(results[key])
            regret_results[key] = optimal_cumulative_reward - cumulative_reward
            
        st.subheader("📊 Visual Proof: Cumulative Regret Comparison (6 Methods)")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for key, regret in regret_results.items():
            ax.plot(regret, label=key, linewidth=1.5)
            
        ax.set_title(f"Simulated Bandit Performance (Averaged over {runs} runs)")
        ax.set_xlabel("Round Index (Budget)")
        ax.set_ylabel("Cumulative Expected Regret")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        st.pyplot(fig)
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env import PortOpsEnv, ActionSpace, MAX_STEPS

def test_score_range():
    env = PortOpsEnv()
    tasks = [1, 2, 3]
    
    for task_id in tasks:
        print(f"\n--- Testing Task {task_id} ---")
        obs = env.reset(task_id=task_id, seed=42)
        done = False
        step = 0
        
        # Simulation loop
        while not done and step < MAX_STEPS:
            # Simple greedy strategy or just enough to finish
            if task_id == 1:
                # Task 1: Unstack and retrieve C01
                if step == 0: cmd = "move(C03, 2)"
                elif step == 1: cmd = "move(C02, 3)"
                else: cmd = "retrieve(C01)"
            elif task_id == 2:
                # Task 2: Place all inbound
                if not obs.inbound_queue: break
                cmd = f"move({obs.inbound_queue[0]}, 1)"
            elif task_id == 3:
                # Task 3: Move one inbound and finish
                cmd = "move(C13, 1)" if step == 0 else "retrieve(C07)"
            
            obs, reward, done, info = env.step(ActionSpace(command=cmd))
            print(f"Step {step}: action={cmd}, reward={reward}, done={done}, score={info.get('score')}")
            
            if done:
                score = info.get('score', 0.0)
                print(f"TERMINAL: score={score}")
                if score <= 0.0 or score >= 1.0:
                    print(f"!! FAILURE: Task {task_id} score {score} is not strictly in (0, 1)")
                else:
                    print(f"OK: Task {task_id} score {score} is in (0, 1)")
                
                # Check meta keys
                required_keys = ["actual_moves", "optimal_moves", "temporal_inversions", "unnecessary_steps"]
                missing = [k for k in required_keys if k not in info]
                if missing:
                    print(f"!! FAILURE: Missing keys from info: {missing}")
                else:
                    print(f"OK: All grading metadata present.")
            
            step += 1

if __name__ == "__main__":
    test_score_range()

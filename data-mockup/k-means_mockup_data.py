import csv
import random
import string
import json

NUM_USERS = 1000
OUTPUT_CSV = "k-means-train-data.csv"
WORK_CATEGORIES = [
    "Research",
    "Writing",
    "Presentation Design",
    "Presenting",
    "Planning",
    "Programming",
    "Graphic Design",
    "Spreadsheet Work",
    "Problem Solving",
    "Content Creation",
    "Script Writing",
    "Reviewing",
    "Documentation",
    "Testing",
    "Report Formatting",
    "Translation",
    "Drawing/Illustration",
    "Code Review",
    "Diagram Creation",
    "Flowchart Design",
    "Mockup Design",
    "Storyboarding",
    "Email Writing",
    "Peer Review",
    "Reference Finding"
]

def random_name():
    return ''.join(random.choices(string.ascii_lowercase, k=5)).capitalize() + str(random.randint(10, 99))

def scale(value, in_min, in_max, out_min, out_max):
    return out_min + ((value - in_min) / (in_max - in_min)) * (out_max - out_min)

def generate_mock_user(user_id):
    num_days = random.randint(3, 20)

    # Quality base score
    quality_base = random.uniform(0, 1)  # 0 = แย่, 1 = ดี

    # Random overall quality score by quality base
    overall_quality_score = round(scale(quality_base, 0, 1, 0, 100) + random.uniform(-3, 3), 2)
    overall_quality_score = max(0, min(overall_quality_score, 100))

    # Random average finish work time (hour) of all project by quality base
    inverse_quality = 1 - quality_base
    base_speed = scale(inverse_quality, 0, 1, 5, 60)  # quality สูง → speed ต่ำ
    work_speed = round(random.gauss(base_speed, 5), 2)
    work_speed = max(0.1, min(work_speed, 72))

    # Random workload per day in int array
    base_workload = scale(quality_base, 0, 1, 3, 10)
    daily_workload = [min(max(int(random.gauss(base_workload, 1.5)), 0), 10) for _ in range(num_days)]

    # Random team_work by quality base
    team_work = round(scale(quality_base, 0, 1, 30, 95) + random.uniform(-5, 5), 2)
    team_work = max(0, min(team_work, 100))

    return {
        "id": user_id,
        "user_name": random_name(),
        "work_load_per_day": daily_workload,
        "team_work": team_work,
        "work_category": random.choice(WORK_CATEGORIES),
        "work_speed": work_speed,
        "overall_quality_score": overall_quality_score
    }

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "id", "user_name", "work_load_per_day", "team_work",
            "work_category", "work_speed", "overall_quality_score"
        ])
        for user in data:
            writer.writerow([
                user["id"],
                user["user_name"],
                json.dumps(user["work_load_per_day"]),
                user["team_work"],
                user["work_category"],
                json.dumps(user["work_speed"]),
                user["overall_quality_score"]
            ])

# Main
if __name__ == "__main__":
    users = [generate_mock_user(i+1) for i in range(NUM_USERS)]
    save_to_csv(users, OUTPUT_CSV)
    print(f"Generated {NUM_USERS} users with consistent trend scores in {OUTPUT_CSV}")

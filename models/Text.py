# ============================================================
# RAER PROMPTS – FULL COLLECTION (Baseline + Variants)
# Author: (you)
# Purpose: Fair comparison & ablation study
# ============================================================


# ============================================================
# 1. BASELINE PROMPTS (GIỮ NGUYÊN – KHÔNG CHỈNH)
# ============================================================

BASELINE = {

    "class_names_5": [
        'Neutrality in learning state.',
        'Enjoyment in learning state.',
        'Confusion in learning state.',
        'Fatigue in learning state.',
        'Distraction.'
    ],

    "class_names_with_context_5": [
        'an expression of Neutrality in learning state.',
        'an expression of Enjoyment in learning state.',
        'an expression of Confusion in learning state.',
        'an expression of Fatigue in learning state.',
        'an expression of Distraction.'
    ],

    # ---- Face only ----
    "class_descriptor_5_only_face": [
        'Relaxed mouth,open eyes,neutral eyebrows,smooth forehead,natural head position.',
        'Upturned mouth,sparkling or slightly squinted eyes,raised eyebrows,relaxed forehead.',
        'Furrowed eyebrows, slightly open mouth, squinting or narrowed eyes, tensed forehead.',
        'Mouth opens in a yawn, eyelids droop, head tilts forward.',
        'Averted gaze or looking away, restless or fidgety posture, shoulders shift restlessly.'
    ],

    # ---- Face + body (with context) ----
    "class_descriptor_5": [
        'Relaxed mouth,open eyes,neutral eyebrows,no noticeable emotional changes,engaged with study materials, or natural body posture.',
        'Upturned mouth corners,sparkling eyes,relaxed eyebrows,focused on course content,or occasionally nodding in agreement.',
        'Furrowed eyebrows, slightly open mouth, wandering or puzzled gaze, chin rests on the palm,or eyes lock on learning material.',
        'Mouth opens in a yawn, eyelids droop, head tilts forward, eyes lock on learning material, or hand writing.',
        'Shifting eyes, restless or fidgety posture, relaxed but unfocused expression,frequently checking phone,or averted gaze from study materials.'
    ]
}


# ============================================================
# 2. STUDENT-CENTRIC PROMPTS (HỌC TẬP – NGỮ CẢNH RÕ)
# ============================================================

STUDENT_CONTEXT = {

    "class_names_5": [
        "Neutral (student in class).",
        "Enjoyment (student in class).",
        "Confusion (student in class).",
        "Fatigue (student in class).",
        "Distraction (student in class)."
    ],

    "class_names_with_context_5": [
        "A student shows a neutral learning state in a classroom.",
        "A student shows enjoyment while learning in a classroom.",
        "A student shows confusion during learning in a classroom.",
        "A student shows fatigue during learning in a classroom.",
        "A student shows distraction and is not focused in a classroom."
    ]
}


# ============================================================
# 3. DESCRIPTOR PROMPTS (FACE / BODY / FACE+BODY)
# ============================================================

DESCRIPTORS = {

    # ---- Face only ----
    "only_face": [
        "A student has a neutral face with relaxed mouth, open eyes, and calm eyebrows.",
        "A student looks happy with a slight smile, bright eyes, and relaxed eyebrows.",
        "A student looks confused with furrowed eyebrows, a puzzled look, and slightly open mouth.",
        "A student looks tired with drooping eyelids, frequent yawning, and a sleepy face.",
        "A student looks distracted with unfocused eyes and a wandering gaze away from the lesson."
    ],

    # ---- Body only ----
    "only_body": [
        "A student sits still with an upright posture and hands on the desk, showing a neutral learning state.",
        "A student leans slightly forward with an open, engaged posture, showing enjoyment in learning.",
        "A student tilts the head and leans in, hand on chin, showing confusion while trying to understand.",
        "A student slouches with shoulders dropped and head lowered, showing fatigue during class.",
        "A student shifts around, turns away from the desk, or looks sideways, showing distraction and low focus."
    ],

    # ---- Face + body ----
    "face_and_body": [
        "A student looks neutral and calm in class, with a relaxed face and steady gaze, quietly watching the lecture or reading notes.",
        "A student shows enjoyment while learning, with a gentle smile and bright eyes, appearing engaged and interested in the lesson.",
        "A student looks confused in class, with furrowed eyebrows and a puzzled expression, focusing on the material as if trying to understand.",
        "A student appears fatigued in class, with drooping eyelids and yawning, head slightly lowered, showing low energy.",
        "A student is distracted in class, frequently looking away from the lesson, scanning around, and not paying attention to learning materials."
    ]
}


# ============================================================
# 4. PROMPT ENSEMBLE (MULTI-PROMPT PER CLASS)
# ============================================================

PROMPT_ENSEMBLE_5 = [
    [   # Neutral
        "A photo of a student with a neutral expression.",
        "A photo of a student sitting still and watching the lecture.",
        "A photo of a student with a calm face and neutral body posture."
    ],
    [   # Enjoyment
        "A photo of a student showing enjoyment while learning.",
        "A photo of a student with a happy face and a slight smile.",
        "A photo of a student who looks engaged and interested in the lesson."
    ],
    [   # Confusion
        "A photo of a student who is confused.",
        "A photo of a student with a puzzled look and furrowed eyebrows.",
        "A photo of a student staring at the material as if trying to understand."
    ],
    [   # Fatigue
        "A photo of a student who appears fatigued or sleepy.",
        "A photo of a student with drooping eyelids or yawning.",
        "A photo of a student showing low energy with a lowered head."
    ],
    [   # Distraction
        "A photo of a student who is distracted from learning.",
        "A photo of a student looking away from the lesson or checking a phone.",
        "A photo of a student with a wandering gaze and unfocused eyes."
    ]
]


# ============================================================
# 5. FACIAL EMOTION PROMPTS (7 & 8 CLASSES – KHÔNG HỌC TẬP)
# ============================================================

EMOTION_7 = {
    "class_names": ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger'],
    "descriptor": [
        'A person who is feeling neutral.',
        'A person who is feeling happy.',
        'A person who is feeling sad.',
        'A person who is feeling surprise.',
        'A person who is feeling fear.',
        'A person who is feeling disgust.',
        'A person who is feeling anger.'
    ]
}

EMOTION_8 = {
    "class_names": ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'],
    "descriptor": [
        'A person who is feeling neutral.',
        'A person who is feeling happy.',
        'A person who is feeling sad.',
        'A person who is feeling surprise.',
        'A person who is feeling fear.',
        'A person who is feeling disgust.',
        'A person who is feeling anger.',
        'A person who is feeling contempt.'
    ]
}
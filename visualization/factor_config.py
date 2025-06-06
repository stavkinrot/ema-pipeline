# factor_config.py

FACTOR_QUESTIONS = {
    "Parental Involvement": {
        "child": ["C_Inv_Help", "C_Inv_Fun", "C_Inv_Talk"],
        "parent": ["P_Inv_Help", "P_Inv_Fun", "P_Inv_Talk"]
    },
    # "Positive Parenting": {
    #     "child": ["C_Positive"],
    #     "parent": ["P_Positive"]
    # },
    "Anxiety": {
        "child": ["C_Anx_Worry", "C_Anx_now"],
        "parent": ["P_Anx_Worry", "P_Anx_now"]
    },
    "Anger": {
        "child": ["C_Agr_NotAsWant"],
        "parent": ["P_Agr_NotAsWant"]
    },
    "Irritability": {
        "child": ["C_Irr_Frustration", "C_Angry_now"],
        "parent": ["P_Irr_Frustration", "P_Angry_now"]
    },
    "Parent-Child Conflict": {
        "child": ["C_PC_Annoy", "C_PC_Criticism"],
        "parent": ["P_PC_Annoy", "P_PC_Criticism"]
    },
    # "Parent-Child Connection": {
    #     "child": ["C_PC_Sharing"],
    #     "parent": ["P_PC_Sharing"]
    # },
    "Depression": {
        "child": ["C_Mood_Sad"],
        "parent": ["P_Mood_Sad"]
    },
    # "Mood - Good": {
    #     "child": ["C_Mood_Good"],
    #     "parent": ["P_Mood_Good"]
    # },
    "ADHD Symptoms": {
        "child": ["C_ADHD_Distracted", "C_ADHD_Restless"],
        "parent": ["P_ADHD_Distracted", "P_ADHD_Restless"]
    },
    "Inhibitory Control": {
        "child": ["C_IC_FirstOnMind", "C_IC_CantStop"],
        "parent": ["P_IC_FirstOnMind", "P_IC_CantStop"]
    },
    "Authoritarian Parenting": {
        "child": ["C_PS_GotAngry"],
        "parent": ["P_PS_GotAngry"],
    },
    "Authoritative Parenting": {
        "child": ["C_PS_Patient"],
        "parent": ["P_PS_Patient"],
    },
    # "Permissive Parenting": {
    #     "child": ["C_PS_Agree"],
    #     "parent": ["P_PS_Agree"],
    # },
}

PALETTE = [
    "#FFB3BA",  #0 pastel red
    "#FFDFBA",  #1 pastel orange
    "#FFFFBA",  #2 pastel yellow
    "#BAFFC9",  #3 pastel green
    "#BAE1FF",  #4 pastel blue
    "#D5BAFF",  #5 pastel violet
    "#FFBAED",  #6 pastel pink
    "#BAFFF0",  #7 aqua-mint
    "#F0BAFF",  #8 light magenta
    "#BAFFB9",  #9 spring green
    "#C2F0FC",  #10 light cyan
    "#FFC2E0",  #11 soft rose
    "#C6FFDD",  #12 lime-mint
    "#E0C2FF",  #13 lavender-purple
    "#FFF2BA",  #14 cream yellow
]

# Semantic mapping for aggression plot
AGGRESSION_COLORS = {
    "none": PALETTE[3],           # mint green (no aggression)
    "some": PALETTE[0],           # pastel red (some aggression)

    "C_Agr_other": PALETTE[5],    # violet
    "C_Agr_slam": PALETTE[1],     # orange
    "C_Agr_throw_smt": PALETTE[11],# red
    "C_Agr_throw_twd": PALETTE[6],# pink
    "C_Agr_yelled": PALETTE[4],   # blue
}

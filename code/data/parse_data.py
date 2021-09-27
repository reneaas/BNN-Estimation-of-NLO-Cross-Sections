infilename = "1_3725_1.slha"

#BLOCKS ins SLHA data.
blocks = {}
#Particle ids
ids = ["1000022", "1000023", "1000024", "1000025", "1000035", "1000037"]

blocks["BLOCK MASS"] = {}

with open(infilename, "r") as infile:
    lines = infile.readlines()

    for line in lines:
        

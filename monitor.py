import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import plotext as pltt




def main(args):
    histoire = pd.read_json(args.file)

    valid_data, train_data = list(histoire["valid"]), list(histoire["train"])
    stoi = list(histoire["stoi"])
    
    if args.epoch == "tv":
        data = [train_data,valid_data]
        legend = ["entraînement","validation"]
    elif args.epoch == "t":
        data = [train_data]
        legend = ["entraînement"]
    elif args.epoch == "v" and len(valid_data) > 0:
        data = [valid_data]        
        legend = ["validation"]
    else:
        print("--epoch (t/v/tv)")

    
    for x,l in zip(data,legend):
        # Print dans le terminal avec plottext
        pltt.scatter(x, label=l)        

        if args.save == 'y':
            # Sauvegarde de plot propres avec matplotlib
            plt.plot(x, label=l)


    title = str("-".join(legend)) + " – " + str(len(valid_data)) + " epochs"

    pltt.xlabel("nb epochs")
    pltt.ylabel("loss")
    pltt.title(title)
    pltt.show()
    if args.save == 'y':
        print("ok")
        plt.title(title)
        plt.xlabel("nombre d'itération")
        plt.ylabel("L1")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, title+".png"))
        plt.clf()





if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", default=None, type=str,
                        required=True, help="Chemin vers history.json")
    parser.add_argument("-e", "--epoch", default='tv', type=str,
                        required=False, help="(t/v/tv) Info sur les epoch train ou valid")
    parser.add_argument("-s", "--save", default='n', type=str,
                        required=False, help="y/n si vous voulez sauvegarder les graphes")
    parser.add_argument("-o", "--output_dir", default='.', type=str,
                        required=False, help="Dossier de sortie des graphes")



    args = parser.parse_args()
    main(args)



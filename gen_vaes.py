from vae_model import *

scenario_dfs = []
for i in range(1, 10):
    scenario_dfs.append(pd.read_csv(f"CSVs/scenario{i}.csv"))

latent_dims = 15
noise = np.load('data/noise.npy')[:, :latent_dims]
noise = torch.from_numpy(noise)

for i, scenario_df in enumerate(scenario_dfs):
    yields_df = scenario_df[["YIELD_1", "YIELD_2", "YIELD_3", "YIELD_4"]]
    #yields_df = (yields_df - yields_df.mean()) / yields_df.std()

    # Load the model
    vae = VariationalAutoencoder(latent_dims, input_dims=4, output_dims=4, verbose=False)
    vae_state_dict_path = f'models/vae{i}_model.pth'
    vae.load_state_dict(torch.load(vae_state_dict_path))

    generator = vae.decoder
    generator.eval()

    # Generate the distribution
    gen_tensor = generator(noise[:len(yields_df)])
    gen = gen_tensor.detach().numpy()

    gen_df = pd.DataFrame(gen, columns=["YIELD_1", "YIELD_2", "YIELD_3", "YIELD_4"])

    print(f"SWD for scenario {i}: ", ot.sliced.sliced_wasserstein_distance(gen_df.to_numpy(), yields_df.to_numpy()))

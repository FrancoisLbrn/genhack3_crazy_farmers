from vae_model import *

scenario_dfs = []
for i in range(1, 10):
    scenario_dfs.append(pd.read_csv(f"CSVs/scenario{i}.csv"))

verbose = False
epochs = 200
lr = 1e-3
latent_dims = 15

for i, scenario_df in enumerate(scenario_dfs):
    print(f"Training VAE #{i+1}...")
    #normalized_df = (scenario_df - scenario_df.mean()) / scenario_df.std()
    scenario_tensor = torch.tensor(scenario_df[["YIELD_1", "YIELD_2", "YIELD_3", "YIELD_4"]].values)

    vae = VariationalAutoencoder(latent_dims=latent_dims, input_dims=4, output_dims=4, verbose=verbose)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)#, weight_decay=1e-3)

    # Train
    # ----------------------------------------------------------
    for epoch in range(epochs):
        train_loss = train_vae(vae,scenario_tensor, scenario_tensor, optimizer)
        torch.cuda.empty_cache()
        if epoch % 10 == 0:
            print('\n EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, epochs,train_loss))

    # SAVE MODEL
    torch.save(vae.state_dict(), f'models/vae{i}_model.pth')
    print(f"Finished training VAE #{i+1}")

print("Done!")
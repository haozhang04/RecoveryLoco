import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rsl_rl.utils import get_activation


class VAEEstimator(nn.Module):
    """VAE-based velocity and latent-state estimator."""

    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        enc_hidden_dims=[128, 64, 32],
        decoder_hidden_dims=[128, 128],
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        kld_weight=1.0,
        **kwargs,
    ):
        if kwargs:
            print("VAE_Estimator.__init__ got unexpected arguments, which will be ignored: "
                  + str(list(kwargs.keys())))
        super().__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.kld_weight = kld_weight

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        self.encoder = nn.Sequential(*enc_layers)

        encoder_output_dim = enc_hidden_dims[-2]
        # Latent 隐变量分布
        self.latent_mean = nn.Linear(encoder_output_dim, self.num_latent)
        self.latent_logvar = nn.Sequential(
            nn.Linear(encoder_output_dim, self.num_latent),
            nn.Hardtanh(min_val=-5.0, max_val=5.0),
        )
        # Velocity 速度分布
        self.vel_mean = nn.Linear(encoder_output_dim, 3)
        self.vel_logvar = nn.Sequential(
            nn.Linear(encoder_output_dim, 3),
            nn.Hardtanh(min_val=-5.0, max_val=5.0),
        )

        # Decoder
        dec_input_dim = self.num_latent + 3
        dec_layers = []
        for l in range(len(decoder_hidden_dims)):
            dec_layers += [nn.Linear(dec_input_dim, decoder_hidden_dims[l]), activation]
            dec_input_dim = decoder_hidden_dims[l]
        dec_layers += [nn.Linear(dec_input_dim, self.num_one_step_obs)]
        self.decoder = nn.Sequential(*dec_layers)

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def encode_history(self, obs_history):
        encoded = self.encoder(obs_history.detach())
        return (
            self.vel_mean(encoded),
            self.vel_logvar(encoded),
            self.latent_mean(encoded),
            self.latent_logvar(encoded),
        )

    def encode(self, obs_history):
        vel_mean, _, latent_mean, _ = self.encode_history(obs_history)
        return vel_mean, latent_mean

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def decode(self, latent, vel):
        return self.decoder(torch.cat((vel, latent), dim=-1))

    def get_latent(self, obs_history):
        vel, latent = self.encode(obs_history)
        return vel.detach(), latent.detach()

    def forward(self, obs_history):
        # The policy uses distribution means to keep action inference deterministic.
        return self.get_latent(obs_history)

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        next_critic_obs = next_critic_obs.detach()
        target_vel = next_critic_obs[:, self.num_one_step_obs : self.num_one_step_obs + 3]
        target_next_obs = next_critic_obs[:, 3:self.num_one_step_obs + 3]

        vel_mean, vel_logvar, latent_mean, latent_logvar = self.encode_history(obs_history)
        sampled_vel = self.reparameterize(vel_mean, vel_logvar)
        sampled_latent = self.reparameterize(latent_mean, latent_logvar)
        reconstructed_obs = self.decode(sampled_latent, sampled_vel)

        estimation_loss = F.mse_loss(vel_mean, target_vel)
        reconstruction_loss = F.mse_loss(reconstructed_obs, target_next_obs)
        kld_loss = -0.5 * torch.mean(
            torch.sum(
                1.0
                + latent_logvar
                - latent_mean.pow(2)
                - latent_logvar.exp(),
                dim=-1,
            )
        )
        vae_loss = reconstruction_loss + self.kld_weight * kld_loss
        loss = estimation_loss + vae_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), vae_loss.item()

import matplotlib.pyplot as plt


def plot_our_sampler_metrics(norm_traj, norm_grad_norms, cond_score_norms, x0_norms, x1_errs, savepath):
    fig, axs = plt.subplots(1, 5, figsize=(5*5,4))

    axs[0].plot(norm_traj, color='k')
    axs[0].set_xlabel("Langevin Steps", fontsize=14)
    axs[0].set_ylabel(r"$\|y - \mathcal{A}(\hat{x}_1)\|$", fontsize=14)
    axs[0].set_yscale('log')

    axs[1].plot(norm_grad_norms, color='k')
    axs[1].set_xlabel("Langevin Steps", fontsize=14)
    axs[1].set_ylabel(r"$\nabla\|y - \mathcal{A}(\hat{x}_1)\|^2$", fontsize=14)
    axs[1].set_yscale('log')

    axs[2].plot(cond_score_norms, color='k')
    axs[2].set_xlabel("Langevin Steps", fontsize=14)
    axs[2].set_ylabel(r"$\rho\nabla\|y - \mathcal{A}(\hat{x}_1)\|^2$", fontsize=14)
    axs[2].set_yscale('log')

    axs[3].plot(x0_norms, color='k')
    axs[3].set_xlabel("Langevin Steps", fontsize=14)
    axs[3].set_ylabel(r"$\|x_0\|$", fontsize=14)
    axs[3].set_yscale('log')

    axs[4].plot(x1_errs, color='k')
    axs[4].set_xlabel("Langevin Steps", fontsize=14)
    axs[4].set_ylabel(r"$\|x_1 - \hat{x}_1\|$", fontsize=14)
    axs[4].set_yscale('log')

    plt.tight_layout()
    plt.savefig(savepath)
    plt.clf()
    plt.close()
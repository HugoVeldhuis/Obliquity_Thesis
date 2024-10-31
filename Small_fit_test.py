
import scipy.stats
import ironman
import pandas as pd
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import rmfit
import corner
import lightkurve as lk

# folder_counter = 1

df = pd.read_csv('TOI1259_RV.csv')

search_result = lk.search_lightcurve('TIC 288735205', mission='TESS')
data_1 = search_result[1].download()



exp_times = {"TESSY1":False,"NEID": 600.0/60.0/60.0/24.0}

time = df['bjd'].values
rv = df['rv'].values
error = df['e_rv'].values



times_lc, fluxes, fluxes_error = {}, {}, {}
times_rvs, rvs, rvs_err =  {}, {}, {}
times_RM, RM, RM_err =  {}, {}, {}

times_RM["NEID"], RM["NEID"], RM_err["NEID"] = time, rv, error

# + 2400000.0
times_lc["TESSY1"], fluxes["TESSY1"], fluxes_error["TESSY1"] = lc["time"].values, lc["flux"].values, lc['flux_err'].values
#times_lc["TESSY2"], fluxes["TESSY2"], fluxes_error["TESSY2"] = bjd_tess2, flux_tess2, flux_err_tess2


output_string = f"including_TESS_data_RM_fit_1"
prior_string = "single_TESS_included_priors.dat"


# if not: make the folder, enter priors and fit
#     folder_counter +=1
#output_string = f"True_obliquities_results_{folder_counter}"

data = ironman.DataOrganizer(output=output_string,     lc_time=times_lc,lc_data=fluxes,lc_err=fluxes_error,rv_time=times_rvs,rv_data=rvs,rv_err=rvs_err,rm_time=times_RM,rm_data=RM,rm_err=RM_err,verbose = True,exp_times=exp_times)
priors = ironman.Priors(prior_string,data)
fit = ironman.Fit(data=data,priors=priors)
# print('checkpoint')
postsamples = fit.run(n_live=500, nthreads = 24)
results = ironman.Results(fit)


# print('New folder and fit created in folder ', output_string)

results.print_mass_radius_rho_sma_planet(r_units=u.Rearth,m_units=u.Mearth)

times_espresso = results.data.x["NEID"]
times_models_espresso = np.linspace(times_espresso[0],times_espresso[-1],10000)
rm_models = results.evaluate_RM_model(times_models_espresso,"NEID",n_models=True,n=10000)
rm_models = rm_models - results.vals["gamma_NEID"]
rm_model_obs = results.evaluate_RM_model(times_espresso,"NEID")



fig = plt.figure(figsize=(8,8),dpi=150,constrained_layout=True)
gs = fig.add_gridspec(2, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex = ax1)


t0 = results.vals["t0_p1"]
P = results.vals["per_p1"]

############################################ RM Effect

jitter = results.vals["sigma_NEID"]
gamma = results.vals["gamma_NEID"]
times_model_plot = (times_models_espresso-t0-556*P)*24.
times, data, error = results.data.x["NEID"],results.data.y["NEID"],np.sqrt(results.data.yerr["NEID"]**2.0 + jitter**2.0)
times = (times-t0-556*P)*24. #556 corresponds to the number of the transit (i.e. the ESPRESSO transit midpoint was at t_i = 556 * per + t0 )
ax1.plot(times_model_plot,np.quantile(rm_models,0.5,axis=0),lw=2,color="crimson",zorder=-10)
ax1.fill_between(times_model_plot,np.quantile(rm_models,0.16,axis=0),np.quantile(rm_models,0.84,axis=0),alpha=0.3,color="#EE2C2C",lw=0,zorder=-1)
ax1.errorbar(times,data-gamma,error,fmt="o",color="k",capsize=2,elinewidth=1,markerfacecolor="dimgrey")
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_ylabel('RV (m/s)',fontsize=11)
ax1.set_title("a) RM Effect", fontsize=14)

res = data-rm_model_obs
ax2.errorbar(times,res,error,fmt="o",color="k",capsize=2,elinewidth=1,markerfacecolor="dimgrey")
ax2.axhline(0.0,lw=2,color="crimson",zorder=-10)
ax2.set_ylim(-15,15)
#ax2.set_xticks([-3,-2,-1,0,1,2])
ax2.set_xlabel(r'Hours from mid-transit',fontsize=11)
ax2.set_ylabel('O - C')

plt.savefig('RM_curve.png')
plt.clf()

fig = plt.figure(figsize=(8,4),dpi=150,constrained_layout=True)
gs = fig.add_gridspec(6, 12)

ax7 = fig.add_subplot(gs[:2, 2:6])
ax8 = fig.add_subplot(gs[2, 2:6], sharex=ax7)
# ax9 = fig.add_subplot(gs[:2, 6:10], sharey=ax7,sharex=ax7)
# ax10 = fig.add_subplot(gs[2, 6:10],sharex=ax7,sharey=ax8)

times_TESSY1 = results.data.x["TESSY1"]
lc_model_TESSY1 = results.evaluate_LC_model(times_TESSY1,"TESSY1")
jitter_TESSY1 = results.vals["sigma_TESSY1"]
phase_TESSY1 = ((times_TESSY1-t0 + 0.5*P) % P)/P
idx = np.argsort(phase_TESSY1)

data_TESSY1, error_TESSY1 = results.data.y["TESSY1"], np.sqrt(results.data.yerr["TESSY1"]**2.0 + jitter_TESSY1**2.0)
ax7.plot((phase_TESSY1[idx]-0.5)*P*24.,lc_model_TESSY1[idx],lw=2,color="crimson",zorder=10)
ax7.errorbar((phase_TESSY1-0.5)*P*24.,data_TESSY1,error_TESSY1,fmt=".",color="cornflowerblue",elinewidth=1)
ax7.set_yticks([0.98,0.99,1,1.01])
ax7.set_xlim(-2.50,2.5)
ax7.set_xticks([-2,-1,0,1,2])
plt.setp(ax7.get_xticklabels(), visible=False)
ax7.set_ylabel('Relative Flux',size=11)
ax7.set_title("Single TESS Transit",fontsize=14)
res_TESSY1 = data_TESSY1-lc_model_TESSY1
ax8.errorbar((phase_TESSY1-0.5)*P*24.,res_TESSY1,error_TESSY1,fmt=".",color="cornflowerblue",elinewidth=1)
ax8.axhline(0.0,lw=2,color="crimson",zorder=10)
#ax8.set_xlim(-3,3)
ax8.set_xlabel(r'Hours from mid-transit',size=11)
ax8.set_ylabel('O - C',size=11)

# times_TESSY2 = results.data.x["TESSY2"]
# lc_model_TESSY2 = results.evaluate_LC_model(times_TESSY2,"TESSY2")
# jitter_TESSY2 = results.vals["sigma_TESSY2"]
# phase_TESSY2 = ((times_TESSY2-t0 + 0.5*P) % P)/P
# idx = np.argsort(phase_TESSY2)
# data_TESSY2, error_TESSY2 = results.data.y["TESSY2"], np.sqrt(results.data.yerr["TESSY2"]**2.0 + jitter_TESSY2**2.0)
# ax9.plot((phase_TESSY2[idx]-0.5)*P*24.,lc_model_TESSY2[idx],lw=2,color="crimson",zorder=10)
# ax9.errorbar((phase_TESSY2-0.5)*P*24.,data_TESSY2,error_TESSY2,fmt=".",color="cornflowerblue",elinewidth=1)
# plt.setp(ax9.get_xticklabels(), visible=False)
# plt.setp(ax9.get_yticklabels(), visible=False)
# ax9.minorticks_off()
# #rmfit.utils.ax_apply_settings(ax9,ticksize=20)
# ax9.set_title("e) TESS Year 3",fontsize=14)
# res_TESSY2 = data_TESSY2-lc_model_TESSY2
# ax10.errorbar((phase_TESSY2-0.5)*P*24.,res_TESSY2,error_TESSY2,fmt=".",color="cornflowerblue",elinewidth=1)
# ax10.axhline(0.0,lw=2,color="crimson",zorder=10)
# ax10.set_xlabel(r'Hours from mid-transit',size=11)
# plt.setp(ax10.get_yticklabels(), visible=False)
# ax10.minorticks_off()

fig.set_facecolor('w')
plt.savefig('Tess_data_fit.png')
plt.close()
plt.clf()


to_corner = results.chain[["lam_p1", "vsini_star", "per_p1", "t0_p1", "rho_star", "b_p1", "p_p1","K_p1"]]
corner.corner(to_corner.values,labels=to_corner.columns,quantiles=[0.16, 0.5, 0.84],label_kwargs={"fontsize": 10},
    show_titles=True,
    title_kwargs={"fontsize": 20})
plt.savefig('corner_plot.png')
plt.close()
plt.clf()

to_corner = results.chain[["lam_p1", "psi_p1"]]
corner.corner(to_corner.values,labels=to_corner.columns,quantiles=[0.16, 0.5, 0.84],label_kwargs={"fontsize": 10},
    show_titles=True,
    title_kwargs={"fontsize": 20})
plt.savefig('small_corner_plot.png')
plt.close()
plt.clf()


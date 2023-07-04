import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.index_tricks import nd_grid
from dedalus import public as de
from useful_functions_int import *



def launch_sim(param_dict):
    # Build bases and domain
    N_grid = param_dict['N_grid']
    x_basis = de.Fourier('x', N_grid, interval=(0, 2*np.pi), dealias=1)
    domain = de.Domain([x_basis], grid_dtype=np.float64)

    # CONTROL SHOULD RESPECT SOME CFL CONDITION for small diffusion! on Peclet number

    # Problems Data
    data = {}
    data['diffusion']   = param_dict['diffusion']
    data['phase_lag']   = param_dict['phase_lag']
    data['T_final']     = param_dict['T_final']
    data['dt'     ]     = param_dict['dt']
    data['domain']      = domain
    data['N_grid']      = param_dict['N_grid']

    # OCP Data
    data['beta']   = param_dict['beta']    # control  cost
    data['alfa_R'] = 1.0       # running  cost
    data['alfa_T'] = 1.0      # terminal cost


    x = domain.grid(0,scales=1)

    # Initial State
    state_field         =  domain.new_field(name="z")
    state_field.set_scales(1)
    state_field['g']    =  1 + 0.1*np.sin(x) + 0.1*np.cos(x) + 0.4*np.sin(2*x) 
    C                   =  state_field.integrate('x')     
    state_field.set_scales(1)  
    state_array         =  state_field['g'] / C['g'][0]


    # Target field
    target_field         =  domain.new_field(name="z")
    target_field.set_scales(1)
    target_field['g']    =  np.exp(-(x-3*np.pi/2)**2 / (0.45**2))
    C                    =  target_field.integrate('x')       
    target_field['g']    =  target_field['g'] / C['g'][0]
    target_field.set_scales(1)
    data['target_array'] = target_field['g']                #adjoint solver uses target array in grid space


    box_constraints  = [-np.inf,np.inf]   # should depend on grid size and dt
    stateHandler   = StateHandler(data,state_array)
    adjointHandler = AdjointHandler(data)
    controlHandler = ControlHandler(data,box_constraints)


    # Basic gradient descent
    max_int = param_dict['max_it']


    J_hist       = []
    grad_J_hist  = []
    failed       = False

    # Solve uncontrolled state dynamics
    nc_control_data = {}
    nc_control_data['t_array'] = controlHandler.control_data['t_array']
    nc_control_data['c_array'] = controlHandler.control_data['c_array']
    nc_u_data = stateHandler.solve(nc_control_data)

    '''
    u_array   = nc_u_data['u_array']
    shift     = int(N_grid/2)
    u_shifted = np.hstack((u_array[shift:,-1],u_array[:shift,-1]))
    data['target_array'] = u_shifted
    '''


    stateHandler   = StateHandler(data,state_array)
    adjointHandler = AdjointHandler(data)
    controlHandler = ControlHandler(data,box_constraints)


    for jj in range(max_int):

        J               = controlHandler.compute_cost(stateHandler)
        grad_J          = controlHandler.compute_gradient(stateHandler,adjointHandler)

        J_hist.append(J)
        grad_J_hist.append(np.linalg.norm(grad_J))

        if failed : 
            break

        print('Iteration : ', jj)
        print('Cost is   : ', J)
        print('Grad norm : ', np.linalg.norm(grad_J))

        failed = controlHandler.update_control(stateHandler)

    print('Cost history : ' , J_hist)
    print('Grad history : ' , grad_J_hist)

    opt_u_data = stateHandler.solve(controlHandler.control_data)
    opt_p_data = adjointHandler.solve(opt_u_data,controlHandler.control_data)
    J               = controlHandler.compute_cost(stateHandler)
    grad_J          = controlHandler.compute_gradient(stateHandler,adjointHandler)
    
    opt_c_data = {
        "t_array" : controlHandler.control_data['t_array'], 
        "c_array" : controlHandler.control_array.T,
        "g_array" : controlHandler.grad_array.T
    }

    J_hist.append(J)
    #grad_J_hist.append(grad_J)

    output_data = {'N_grid':N_grid,'target':target_field['g'],'opt_c':opt_c_data , 'nc_u':nc_u_data,'opt_u':opt_u_data,'opt_p':opt_p_data,'J_hist':J_hist,'gJ_hist':grad_J_hist}

    return output_data


def refine_field(domain,field_array,scale):
    # Given field array as space-time grid, increase spacial resolution

    try:
        N_x,N_t = np.shape(field_array)
    except:
        N_x = np.shape(field_array)
        N_t = 1
        field_array = field_array.reshape((N_x[0],1))

    ref_field = []  # refined field list

    for tt in range(N_t):

        temp_field = domain.new_field(name='tmp')
        temp_field['g'] = field_array[:,tt]
        temp_field.set_scales(scale)
        ref_field.append(temp_field['g'])
    
    ref_field_array = np.array(ref_field).T
    fine_grid       = domain.grid(0,scales=scale)

    return fine_grid , ref_field_array


def save_output_fig(name_folder,output_data):

    # unpack dictionary
    nc_u_data    = output_data['nc_u']
    opt_u_data   = output_data['opt_u']
    opt_p_data   = output_data['opt_p']
    opt_c_data   = output_data['opt_c']
    J_hist       = output_data['J_hist']
    grad_J_hist  = output_data['gJ_hist'] 
    target_array = output_data['target']

    N_grid = output_data['N_grid']
    x_basis = de.Fourier('x', N_grid, interval=(0, 2*np.pi), dealias=1)
    domain = de.Domain([x_basis], grid_dtype=np.float64)
    x = domain.grid(0)


    plt.rc('text',usetex = True)
    from matplotlib import cm
    from matplotlib import ticker

    def nice_contourf(domain,t_array,data):

        figN,axN            = plt.subplots()
        fine_grid,fine_data = refine_field(domain,data,10)
        try:
            levels   = np.linspace(np.min(fine_data),np.max(fine_data),300)
            contourf_ = axN.contourf(fine_grid, t_array,fine_data.T,levels=levels)
        except:
            contourf_ = axN.contourf(fine_grid, t_array,fine_data.T)

        cbar = figN.colorbar(contourf_)
        #axN.yaxis.set_major_locator(plt.NullLocator()) # remove y axis ticks
        #axN.xaxis.set_major_locator(plt.NullLocator()) # remove x axis ticks
        cbar.ax.set_yticklabels(["%.2f" % elem for elem in cbar.ax.get_yticks().tolist()])
        cbar.ax.locator_params(nbins=5)
        axN.set_ylabel(r'$t \, [s]$',fontsize=15)
        axN.set_xlabel(r'$\theta \, [rad] $',fontsize=15)
    
        return figN,axN

    '''
    fig_v,ax_v = nice_contourf(domain,nc_u_data['t_array'], nc_u_data['v_array'])
    ax_v.set_title(r'$\textrm{Uncontrolled - }  w[q](\theta,t)$',fontsize=20)
    fig_v.savefig(name_folder + '/unc_v.png')   # save the figure to file


    fig_nc,ax_nc = nice_contourf(domain,nc_u_data['t_array'], nc_u_data['u_array'])
    ax_nc.set_title(r'$\textrm{Uncontrolled - }  q(\theta,t)$',fontsize=20)
    fig_nc.savefig(name_folder+'/Unc_State.png')   # save the figure to file


    
    # Plot solution
    fig,ax =  nice_contourf(domain,opt_u_data['t_array'], opt_u_data['u_array'])
    ax.set_title(r'$\textrm{Controlled - } q(\theta,t)$',fontsize=20)
    fig.savefig(name_folder+'/State.png')   # save the figure to file
    '''
    '''

    fig_vc,ax_vc = nice_contourf(domain,opt_u_data['t_array'], opt_u_data['v_array'])
    ax_vc.set_title(r'$\textrm{Controlled - } w[q](\theta,t)$',fontsize=20)
    fig_vc.savefig(name_folder+'/VC_State.png')   # save the figure to file


    fig1,ax1 = nice_contourf(domain,opt_p_data['t_array'], opt_p_data['p_array'])
    ax1.set_title(r'$\textrm{Adjoint - } p(\theta,t)$',fontsize=20)
    fig1.savefig(name_folder+'/Adjoint.png') 




    opt_u_data['t_array'], opt_u_data['v_array']
    fig2,ax2  = nice_contourf(domain,opt_c_data['t_array'], opt_c_data['c_array'].T)
    ax2.set_title(r'$\textrm{Control - } u_2(\theta,t)$',fontsize=20)
    fig2.savefig(name_folder+'/Control.png') 

    '''

    # New Plot

    t_c             = opt_c_data['t_array'] 

    from scipy import interpolate
    v_t             = interpolate.interp1d(opt_u_data['t_array'],opt_u_data['v_array'],axis=1,bounds_error=False,
                                               fill_value=(opt_u_data['v_array'][:,0],opt_u_data['v_array'][:,1]))
        
    wq = v_t(t_c).T * opt_c_data['c_array']

    figw,axw  = nice_contourf(domain,opt_c_data['t_array'], wq.T)
    axw.set_title(r'$\textrm{Nonlocal transport field - } w[q]\,u_2(\theta,t)$',fontsize=20)
    figw.savefig(name_folder+'/wq.png') 

    '''
    figN,axN = nice_contourf(domain,opt_u_data['t_array'],opt_u_data['u_array'])
    axN.set_title(r'$\textrm{Density - } q(\theta,t)$',fontsize=20)
    figN.savefig(name_folder+'/NEW.png') 

    fig4,ax4  = nice_contourf(domain,opt_c_data['t_array'],opt_c_data['g_array'].T)
    ax4.set_title(r'$\textrm{Reduced Gradient - } \nabla J(\theta,t)$',fontsize=20)
    fig4.savefig(name_folder+'/Gradient.png')


    fig3,ax3 = plt.subplots()

    fine_grid,opt_u_array = refine_field(domain,opt_u_data['u_array'],10)
    ax3.plot(fine_grid,opt_u_array[:,-1],'b',label=r'$q_c(\theta,T)$')

    #ax3.plot(x,opt_u_data['u_array'][:,-1],'b',label=r'$q_c(\theta,T)$')

    fine_grid,nc_u_array = refine_field(domain,nc_u_data['u_array'],10)
    ax3.plot(fine_grid,nc_u_array[:,-1],'k--',label=r'$q(\theta,T)$')

    fine_grid,fine_target_array = refine_field(domain,target_array,10)
    ax3.plot(fine_grid,fine_target_array[:,0],'r',label=r'$z(\theta,T)$')


    ax3.set_ylabel(r'$\textrm{Density} \, \, [\frac{1}{rad}]$',fontsize=15)
    ax3.set_xlabel(r'$\theta \, [rad] $',fontsize=15)
    ax3.set_title(r'$\textrm{Final Time Slice} $',fontsize=20)
    ax3.legend()
    ax3.grid('minor')
    fig3.savefig(name_folder+'/Final.png')     


    fig_J,ax_J = plt.subplots()
    ax_J.plot(J_hist)
    ax_J.grid('minor')
    ax_J.set_xlabel(r'$\textrm{Iterations}$',fontsize=15)
    ax_J.set_ylabel(r'$\textrm{Cost} $',fontsize=15)
    ax_J.set_title(r'$\textrm{Cost Convergence} $',fontsize=20)
    fig_J.savefig(name_folder+'/conv_cost.png')   # save the figure to file

    fig_G,ax_G = plt.subplots()
    ax_G.plot(grad_J_hist)
    ax_G.grid('minor')
    ax_G.set_xlabel(r'$\textrm{Iterations}$',fontsize=15)
    ax_G.set_ylabel(r'$\textrm{Gradient Norm} $',fontsize=15)
    ax_G.set_title(r'$\textrm{Gradient Convergence} $',fontsize=20)
    fig_G.savefig(name_folder+'/conv_grad.png')   # save the figure to file

    fig_M,ax_M = plt.subplots()
    ax_M.plot(opt_u_data['m_array'])
    ax_M.plot(nc_u_data['m_array'],'--')

    ax_M.grid('minor')
    ax_M.set_xlabel(r'$t \, [s]$',fontsize=15)
    ax_M.set_ylabel(r'$ [-] $',fontsize=15)
    ax_M.set_title(r'$\textrm{Mass conservation} $',fontsize=20)
    fig_M.savefig(name_folder+'/mass_cons.png')   # save the figure to file


    fig_R,ax_R = plt.subplots()
    ax_R.plot(opt_u_data['t_array'],np.abs(opt_u_data['R_array']),label=r'$\textrm{Controlled}$')
    ax_R.plot(nc_u_data['t_array'],np.abs(nc_u_data['R_array']),'--',label=r'$\textrm{Uncontrolled}$')
    ax_R.legend()
    ax_R.set_xlabel(r'$t \, [s]$',fontsize=15)
    ax_R.set_ylabel(r'$R \, [-]$',fontsize=15)
    ax_R.set_title(r'$\textrm{Phase Coherence}$',fontsize=20)
    ax_R.grid('minor')
    fig_R.savefig(name_folder+'/polar_order.png')   # save the figure to file

    fig_p,ax_p = plt.subplots()
    ax_p.plot(opt_u_data['t_array'],np.angle(opt_u_data['R_array']),label=r'$\textrm{Controlled}$')
    ax_p.plot(nc_u_data['t_array'],np.angle(nc_u_data['R_array']),'--',label=r'$\textrm{Uncontrolled}$')
    ax_p.legend()
    ax_p.set_xlabel(r'$t \, [s]$',fontsize=15)
    ax_p.set_ylabel(r'$\psi \, [rad]$',fontsize=15)
    ax_p.set_title(r'$\textrm{Mean Phase}$',fontsize=20)
    ax_p.grid('minor')
    fig_p.savefig(name_folder+'/mean_phase.png')   # save the figure to file

    '''
    #plt.show(block=False)

    return

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg.decomp_qr import qr_multiply
from dedalus import public as de
from scipy import interpolate



def plot_triple(x,t_array,target_field,u_array,p_array,controlHandler):


    plt.rc('text',usetex = True)

    # Plot solution
    fig,ax = plt.subplots()
    contourf_ = ax.contourf(x, t_array, u_array.T)
    cbar = fig.colorbar(contourf_)
    ax.set_ylabel(r'$t \, [s]$',fontsize=15)
    ax.set_xlabel(r'$\theta \, [rad] $',fontsize=15)
    ax.set_title(r'$\textrm{State - } q(\theta,t)$',fontsize=20)
    fig.savefig('State.png')   # save the figure to file

    fig1,ax1 = plt.subplots()
    levels   = np.linspace(np.min(p_array),np.max(p_array),30)
    contourf_ = ax1.contourf(x, t_array, p_array.T,levels)
    cbar = fig1.colorbar(contourf_)
    ax1.set_ylabel(r'$t \, [s]$',fontsize=15)
    ax1.set_xlabel(r'$\theta \, [rad] $',fontsize=15)
    ax1.set_title(r'$\textrm{Adjoint - } p(\theta,t)$',fontsize=20)
    fig1.savefig('Adjoint.png') 

    fig2,ax2  = plt.subplots()
    contourf_ = ax2.contourf(x, controlHandler.control_data['t_array'], controlHandler.control_array.T)
    cbar = fig2.colorbar(contourf_)
    ax2.set_ylabel(r'$t \, [s]$',fontsize=15)
    ax2.set_xlabel(r'$\theta \, [rad] $',fontsize=15)
    ax2.set_title(r'$\textrm{Control - } u(\theta,t)$',fontsize=20)
    fig2.savefig('Control.png') 

    fig3,ax3 = plt.subplots()
    ax3.plot(x,u_array[:,-1],'b',label=r'$q(\theta,T)$')
    ax3.plot(x,target_field['g'],'r',label=r'$z(\theta,T)$')
    ax3.set_ylabel(r'$\textrm{Density} \, \, [\frac{1}{rad}]$',fontsize=15)
    ax3.set_xlabel(r'$\theta \, [rad] $',fontsize=15)
    ax3.set_title(r'$\textrm{Final Time Slice} $',fontsize=20)
    ax3.legend()
    ax3.grid('minor')
    fig3.savefig('Final.png')     

    fig4,ax4  = plt.subplots()
    grad_array = controlHandler.grad_array.T
    levels   = np.linspace(np.min(grad_array),np.max(grad_array),30)
    contourf_ = ax4.contourf(x, controlHandler.control_data['t_array'], grad_array,levels)
    cbar = fig4.colorbar(contourf_)
    ax4.set_ylabel(r'$t \, [s]$',fontsize=15)
    ax4.set_xlabel(r'$\theta \, [rad] $',fontsize=15)
    ax4.set_title(r'$\textrm{Reduced Gradient - } \nabla J(\theta,t)$',fontsize=20)
    fig4.savefig('Gradient.png')

    #plt.show(block=False)

    return

class ControlHandler():

    def __init__(self,data,box_constraints):

        self.data = data
        domain    = data['domain']

        N_t = int(data['T_final']/data['dt'])
        N_x = domain.grid(0,scales=1).shape[0]

        t_array = np.linspace(0,data['T_final']+data['dt'],N_t)     #FIXME please!

        self.control_array = 1*np.ones((N_x,N_t))
        self.grad_field    = np.copy(self.control_array)

        self.control_data = {}
        self.control_data['t_array']       = t_array
        self.control_data['c_array'] = self.control_array

        self.box_constraints = box_constraints

        return

    def set_control(self,control_array):

        self.control_array           = control_array
        self.control_data['c_array'] = control_array

        return

    def compute_cost(self,stateHandler):

        u_data = stateHandler.solve(self.control_data)

        u_array = u_data['u_array']
        t_array = u_data['t_array']

        target_array    = self.data['target_array']
        domain          = self.data['domain']
        dt              = self.data['dt']

        N_x , N_t  = np.shape(u_array)

        target_rep = np.tile(target_array,(N_t,1)).T

        delta_u = domain.new_field(name="du")

        delta_u_t = np.trapz( u_array - target_rep , x = t_array , dx = dt , axis=1)
        delta_u['g'] = delta_u_t 

        delta_u_T = domain.new_field(name="duT")
        delta_u_T['g'] = u_array[:,-1] - target_array

        control_field_t      =  domain.new_field(name="c")
        control_field_t['g'] = np.trapz( self.control_array , x = self.control_data['t_array'] , dx = dt , axis=1) 

        alfa_R = self.data['alfa_R']
        alfa_T = self.data['alfa_T']
        beta   = self.data['beta']

        J     = 0.5 * de.operators.integrate(  alfa_T*delta_u_T**2 + alfa_R*delta_u**2 + beta*control_field_t**2 , 'x' )

        # Compare with Scipy numerical double integral in the grid [0,2\pi] x [0,T_final]
        #from scipy.integrate import simps
        #delta_q_num = 0.5*alfa_R*(u_array - target_rep)**2

        return J['g'][0]


    def compute_gradient(self,stateHandler,adjointHandler):

        u_data = stateHandler.solve(self.control_data)
        p_data = adjointHandler.solve(u_data,self.control_data)

        px_t            = interpolate.interp1d(p_data['t_array'],p_data['px_array'],axis=1,bounds_error=False,
                                               fill_value=(p_data['px_array'][:,0],p_data['px_array'][:,1]))

        u_t             = interpolate.interp1d(u_data['t_array'],u_data['u_array'],axis=1,bounds_error=False,
                                               fill_value=(u_data['u_array'][:,0],u_data['u_array'][:,1]))
        t_c             = self.control_data['t_array']   


        v_t             = interpolate.interp1d(u_data['t_array'],u_data['v_array'],axis=1,bounds_error=False,
                                               fill_value=(u_data['v_array'][:,0],u_data['v_array'][:,1]))
        
        beta            = self.data['beta']
        self.grad_array = beta*self.control_array + px_t(t_c)*v_t(t_c)*u_t(t_c)

        return self.grad_array

    def update_control(self,stateHandler):

        grad_array    = self.grad_array
        tau           = 100
        control_array = self.control_array
        grad_array    = self.grad_array
        J             = self.compute_cost(stateHandler)
        u_min,u_max   = self.box_constraints

        # normalize gradient
        d_k = -(grad_array / np.linalg.norm(grad_array))
        control_array_tmp = control_array + tau*d_k


        # Project on the box
        control_array_tmp = np.clip(control_array_tmp, u_min,u_max)
        self.set_control(control_array_tmp)
        J_tmp   = self.compute_cost(stateHandler)

        max_int = 20
        from math import isnan

        for ii in range(max_int):

            print('J tmp is : ', J_tmp)
            print('tau is : ',tau)

            if J_tmp < J and isnan(J_tmp)==False:

                break

            else : 

                tau = 0.5*tau
                control_array_tmp = control_array + tau*d_k
                # Project on the box
                control_array_tmp = np.clip(control_array_tmp, u_min,u_max)
                self.set_control(control_array_tmp)
                J_tmp   = self.compute_cost(stateHandler)

        failed = False
        if ii == max_int-1:
            failed = True

        return failed


class AdjointHandler():

    def __init__(self,data):

        self.data = data        # data dictionary


    def solve(self,u_data,control_data):


        # Build problem
        domain        = self.data['domain']
        adj_problem = de.IVP(domain, variables=['p','px'])

        adj_problem.parameters['D'] = self.data['diffusion']       #diffusion coefficient
        adj_problem.parameters['a'] = self.data['phase_lag']       #velocity  drift
        a       = self.data['phase_lag']
        x       = domain.grid(0)

        dt      = self.data['dt']
        T_final = self.data['T_final']


        # interpolate in time adjoint forcing   u - z
        u_array_flipped = np.flip(u_data['u_array'],axis=1)
        t_array       = u_data['t_array']

        target_array  = self.data['target_array']
        N_t           = t_array.shape[0]
        target_rep    = np.tile(target_array,(N_t,1)).T
        adjRHS_array  = u_array_flipped - target_rep
        adjRHS_t      = interpolate.interp1d(t_array,adjRHS_array,axis=1,bounds_error = False,fill_value=(adjRHS_array[:,0],adjRHS_array[:,-1]))

        u_t               = interpolate.interp1d(t_array,u_array_flipped,axis=1,bounds_error = False,fill_value=(u_array_flipped[:,0],u_array_flipped[:,-1]))
        u_field_tmp       = domain.new_field(name="u")

        c_field_tmp       = domain.new_field(name="c")

        v_array_flipped   = np.flip(u_data['v_array'],axis=1)
        v_t               = interpolate.interp1d(t_array,v_array_flipped,axis=1,bounds_error=False,fill_value=(v_array_flipped[:,0],v_array_flipped[:,-1]))

        alfa_R  = self.data['alfa_R']  # Running cost


        # interpolate in time control field 
        t_array       = control_data['t_array']
        control_array = control_data['c_array']

        c_t           = interpolate.interp1d(t_array,control_array,axis=1)


        # feed control field to RHS of state equation as general function operator
        def control_t(*args):

            # access simulation time
            t = solver.sim_time

            return c_t(t)

        def control_op(*args, domain=domain, C=control_t):

            return de.operators.GeneralFunction(domain, layout='g', func=C, args=args)

        de.operators.parseables['C'] = control_op




        # feed adjoint forcing to RHS of adjoint equation as general function operator
        def adjRHS_fun(*args):

            # access simulation time
            t = solver.sim_time

            return alfa_R*adjRHS_t(t)

        def adjRHS_op(*args, domain=domain, AdjRHS=adjRHS_fun):

            return de.operators.GeneralFunction(domain, layout='g', func=AdjRHS, args=args)

        de.operators.parseables['AdjRHS'] = adjRHS_op

        # w^{*}[u,q,p_\theta] 
        def r_fun(*args):

            # access simulation time
            t   = solver.sim_time

            # access adjoint derivative
            px  = solver.state['px']

            # define state field for instant considered
            u   = u_t(t)

            # define control field for instant considered
            c   = c_t(t)
            c_field_tmp['g'] = c

            u_field_tmp['g'] = u * np.cos(x+a)

            integ_1 = de.operators.integrate( c_field_tmp*px*u_field_tmp, 'x')
            u_field_tmp['g'] = u * np.sin(x+a)

            integ_2 = de.operators.integrate( c_field_tmp*px*u_field_tmp, 'x')

            out     = np.sin(x)*integ_1['g'] - np.cos(x)*integ_2['g']
            return out

        def r_op(*args, domain=domain, R=r_fun):

            return de.operators.GeneralFunction(domain, layout='g', func=R, args=args)

        de.operators.parseables['R'] = r_op



        # v convolution term from state equation (FLIPPED IN TIME)
        def v_fun(*args):

            # access solver time
            t = solver.sim_time

            return v_t(t)

        def v_op(*args, domain=domain, V=v_fun):

            return de.operators.GeneralFunction(domain, layout='g', func=V, args=args)

        de.operators.parseables['V'] = v_op



        # Add main equation, with linear terms on the LHS and nonlinear terms on the RHS
        adj_problem.add_equation("dt(p) - D*dx(px) =  C()*V()*px +R() +  AdjRHS()")

        # Add auxiliary equation defining the first-order reduction
        adj_problem.add_equation("px - dx(p) = 0")

        # Build solver
        solver = adj_problem.build_solver('RK222')


        # Stopping criteria
        solver.sim_time      = 0
        solver.stop_sim_time = T_final
        solver.stop_wall_time = np.inf
        solver.stop_iteration = np.inf


        # Reference local grid and state fields
        p = solver.state['p']
        px = solver.state['px']

        # Setup a sine wave
        p.set_scales(1)             # Test what happens with N small and high resulotion

        # Adjoint Final condition
        alfa_T = self.data['alfa_T']
        p['g'] = alfa_T*adjRHS_t(0)            # Note: time is already reversed
        p.differentiate('x', out=px)

        # Setup storage
        p.set_scales(1)
        p_list   = [np.copy(p['g'])]
        px_list  = [np.copy(px['g'])]
        t_list   = [solver.sim_time]

        while solver.ok:
            solver.step(dt)
            if solver.iteration % 1 == 0:
                p.set_scales(1)
                p_list.append(np.copy(p['g']))
                px_list.append(np.copy(px['g']))
                t_list.append(solver.sim_time)


        # Convert storage lists to arrays
        p_array  = np.array(p_list[::-1])
        px_array = np.array(px_list[::-1])
        t_array = np.array(t_list)


        p_data = {}
        p_data['t_array']  = t_array
        p_data['p_array']  = p_array.T
        p_data['px_array'] = px_array.T


        return p_data

class StateHandler():

    def __init__(self,data,state_0):

        self.data    = data        # data dictionary
        self.state_0 = state_0

        # define attributes to compute polar order, a complex grid is needed
        self.x_cbasis    =  de.Fourier('y', data['N_grid'], interval=(0, 2*np.pi), dealias=1)
        self.cdomain     =  de.Domain([self.x_cbasis], grid_dtype=np.complex128)
        self.polar_order =  self.cdomain.new_field(name="r")
        self.y           =  self.cdomain.grid(0,scales=1)


    def get_polar_order(self,u_num):
        
        self.polar_order.set_scales(1)
        self.polar_order['g'] =  np.exp(1j*self.y) * u_num
        Polar_order      = self.polar_order.integrate('y')
        Polar_order_num  = Polar_order['g'][0]

        return Polar_order_num

    def solve(self,control_data):


        # Build problem
        domain        = self.data['domain']
        state_problem = de.IVP(domain, variables=['u', 'ux','v'])

        state_problem.parameters['D'] = self.data['diffusion']     #diffusion coefficient
        state_problem.parameters['a'] = self.data['phase_lag']         #velocity  drift

        dt      = self.data['dt']
        T_final = self.data['T_final']

        # interpolate in time control field 
        t_array       = control_data['t_array']
        control_array = control_data['c_array']
        c_t                     = interpolate.interp1d(t_array,control_array,axis=1)


        # feed control field to RHS of state equation as general function operator
        def control_t(*args):

            # access simulation time
            t = solver.sim_time

            return c_t(t)

        def control_op(*args, domain=domain, C=control_t):

            return de.operators.GeneralFunction(domain, layout='g', func=C, args=args)

        de.operators.parseables['C'] = control_op


        state_problem.add_equation("v  = -sin(x)*integ(cos(x-a)*u)+cos(x)*integ(sin(x-a)*u)")
        state_problem.add_equation("dx(u) - ux = 0")
        state_problem.add_equation("dt(u) - D*dx(ux) = - dx(C()*v*u) ")

        # Build solver
        solver = state_problem.build_solver('RK222')


        # Stopping criteria
        solver.stop_sim_time = T_final
        solver.stop_wall_time = np.inf
        solver.stop_iteration = np.inf


        # Reference local grid and state fields
        x  = domain.grid(0)
        u  = solver.state['u']
        v  = solver.state['v']
        ux = solver.state['ux']

        # Set state initial conditions
        u.set_scales(1)
        u['g'] = self.state_0
        #u.differentiate('x', out=ux);


        # Define mass integral
        mass_array = []
        m = u.integrate('x')
        mass_array.append( m['g'][0] ) 

        # Setup storage
        u.set_scales(1)
        v.set_scales(1)

        u_list = [np.copy(u['g'])]
        v_list = [np.copy(v['g'])]
        R_list = [self.get_polar_order(u['g'])]
        t_list = [solver.sim_time]

        while solver.ok:
            solver.step(dt)

            u.set_scales(1)
            v.set_scales(1)
            # compute probability density mass
            mass_array.append( m['g'][0] ) 

            u_list.append(np.copy(u['g']))
            R_list.append(self.get_polar_order(u['g']))
            v_list.append(np.copy(v['g']))
            t_list.append(solver.sim_time)



        # Convert storage lists to arrays
        u_array = np.array(u_list)
        R_array = np.array(R_list)
        v_array = np.array(v_list)
        t_array = np.array(t_list)



        # Fill dictionary with data for easier handling
        u_data = {}
        u_data['t_array'] = t_array
        u_data['u_array'] = u_array.T
        u_data['v_array'] = v_array.T
        u_data['R_array'] = R_array
        u_data['m_array'] = mass_array

        return u_data

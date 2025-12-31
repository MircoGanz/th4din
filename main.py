from thdin import *
import numpy as np
from CoolProp.CoolProp import PropsSI
from multiprocessing.dummy import freeze_support


def evap_model(evap):

    N = 50
    k = 1000

    A = evap.parameter['A'].value

    hi_h = evap.ports['in_af'].h.value
    pi_h = evap.ports['in_af'].p.value
    m_h = evap.ports['in_af'].m.value
    hot_fluid = evap.ports['in_af'].fluid

    hi_c = evap.ports['in_r'].h.value
    pi_c = evap.ports['in_r'].p.value
    m_c = evap.ports['in_r'].m.value
    cold_fluid = evap.ports['in_r'].fluid

    def res(x):
        h_h = np.zeros(N)
        h_c = np.zeros(N)
        h_c[0] = hi_c
        h_h[0] = x[0] * 1e5
        for i in range(N - 1):
            Th = PropsSI('T', 'H', h_h[i], 'P', pi_h, hot_fluid)
            Tc = PropsSI('T', 'H', h_c[i], 'P', pi_c, cold_fluid)
            h_h[i + 1] = h_h[i] + k * A / N * (Th - Tc) / m_h
            h_c[i + 1] = h_c[i] + k * A / N * (Th - Tc) / m_c
        return np.array([(h_h[-1] - hi_h) * 1e-5])

    Ti_c = PropsSI('T', 'H', hi_c, 'P', pi_c, cold_fluid)
    ho_h_init = PropsSI('H', 'P', pi_h, 'T', max(Ti_c, 257.0), hot_fluid) * 1e-5
    sol = scipy.optimize.root(res, np.array([ho_h_init]), method='hybr', tol=1e-5)

    if sol.success:
        ho_h = sol.x[0] * 1e5
        h_h = np.zeros(N)
        h_c = np.zeros(N)
        h_c[0] = hi_c
        h_h[0] = ho_h
        for i in range(N - 1):
            Th = PropsSI('T', 'H', h_h[i], 'P', pi_h, hot_fluid)
            Tc = PropsSI('T', 'H', h_c[i], 'P', pi_c, cold_fluid)
            h_h[i + 1] = h_h[i] + k * A / N * (Th - Tc) / m_h
            h_c[i + 1] = h_c[i] + k * A / N * (Th - Tc) / m_c
    else:
        raise Exception(sol.message)

    po_h = pi_h
    ho_h = h_h[0]
    mo_h = m_h

    po_c = pi_c
    ho_c = h_c[-1]
    mo_c = m_c

    evap.ports['out_af'].p.set_value(po_h)
    evap.ports['out_af'].h.set_value(ho_h)
    evap.ports['out_af'].m.set_value(mo_h)

    evap.ports['out_r'].p.set_value(po_c)
    evap.ports['out_r'].h.set_value(ho_c)
    evap.ports['out_r'].m.set_value(mo_c)

    evap.outputs["Q"].set_value(m_c * (ho_c - hi_c))


def comp_model(comp):

    n = 4 # number of cylinders
    bore = 0.06 # bore [m]
    stroke = 0.042 # stroke [m]
    Vd = n * bore**2 * np.pi / 4 * stroke # displacement volume

    a = np.array([-5.31166292e-02, 1.21402922e-03, 8.81226071e-05, 1.03163725e+00])
    b = np.array([9.38116126e-03, -1.52858792e-03, -4.08026601e-03, 6.31332600e-04, 6.77625196e-01])

    RPM = comp.parameter['RPM'].value
    hi = comp.ports['in'].h.value
    pi = comp.ports['in'].p.value
    po = comp.ports['out'].p.value
    fluid = comp.ports['in'].fluid

    eta_v = a[0] * (po / pi) + a[1] * (po / pi) ** 2 + a[2] * (RPM / 60) + a[3]
    eta_is = b[0] * (po / pi) + b[1] * (po / pi) ** 2 + b[2] * (RPM / 60) + b[3] * (po / pi) * (RPM / 60) + b[4]

    s = PropsSI('S', 'P', pi, 'H', hi, fluid)
    h_is = PropsSI('H', 'P', po, 'S', s, fluid)
    rho = PropsSI('D', 'P', pi, 'H', hi, fluid)
    ho = hi + (h_is - hi) / eta_is
    mi = mo = RPM / 60 * Vd * eta_v * rho

    comp.ports['in'].m.set_value(mi)
    comp.ports['out'].h.set_value(ho)
    comp.ports['out'].m.set_value(mo)


def cond_model(cond):

    N = 50
    k = 1000

    A = cond.parameter['A'].value

    hi_h = cond.ports["in_r"].h.value
    pi_h = cond.ports["in_r"].p.value
    m_h = cond.ports["in_r"].m.value
    hot_fluid = cond.ports["in_r"].fluid

    hi_c = cond.ports["in_af"].h.value
    pi_c = cond.ports["in_af"].p.value
    m_c = cond.ports["in_af"].m.value
    cold_fluid = cond.ports["in_af"].fluid

    def res(x):
        h_h = np.zeros(N)
        h_c = np.zeros(N)
        h_h[0] = hi_h
        h_c[0] = x[0] * 1e5
        for i in range(N - 1):
            Th = PropsSI('T', 'H', h_h[i], 'P', pi_h, hot_fluid)
            Tc = PropsSI('T', 'H', h_c[i], 'P', pi_c, cold_fluid)
            h_h[i + 1] = h_h[i] - k * A / N * (Th - Tc) / m_h
            h_c[i + 1] = h_c[i] - k * A / N * (Th - Tc) / m_c
        return np.array([(h_c[-1] - hi_c) * 1e-5])

    Ti_h = PropsSI('T', 'H', hi_h, 'P', pi_h, hot_fluid)
    ho_c_init = PropsSI('H', 'P', pi_c, 'T', min(Ti_h, 350.0), cold_fluid) * 1e-5
    sol = scipy.optimize.root(res, np.array([ho_c_init]), method='hybr', tol=1e-5)

    if sol.success:
        ho_c = sol.x[0] * 1e5
        h_h = np.zeros(N)
        h_c = np.zeros(N)
        h_h[0] = hi_h
        h_c[0] = ho_c
        for i in range(N - 1):
            Th = PropsSI('T', 'H', h_h[i], 'P', pi_h, hot_fluid)
            Tc = PropsSI('T', 'H', h_c[i], 'P', pi_c, cold_fluid)
            h_h[i + 1] = h_h[i] - k * A / N * (Th - Tc) / m_h
            h_c[i + 1] = h_c[i] - k * A / N * (Th - Tc) / m_c
    else:
        raise Exception(sol.message)

    po_h = pi_h
    ho_h = h_h[-1]
    mo_h = m_h

    po_c = pi_c
    ho_c = h_c[0]
    mo_c = m_c

    cond.ports["out_r"].p.set_value(po_h)
    cond.ports["out_r"].h.set_value(ho_h)
    cond.ports["out_r"].m.set_value(mo_h)

    cond.ports["out_af"].p.set_value(po_c)
    cond.ports["out_af"].h.set_value(ho_c)
    cond.ports["out_af"].m.set_value(mo_c)


def exp_model(exp):

    Cf = exp.parameter['Cf'].value
    Av = exp.parameter['Av'].value
    hi = exp.ports["in"].h.value
    pi = exp.ports["in"].p.value
    po = exp.ports["out"].p.value
    fluid = exp.ports["in"].fluid

    rho = PropsSI('D', 'P', pi, 'H', hi, fluid)
    pressure_difference = (pi - po) * 1e-5
    n = pressure_difference * 1000 * rho
    m = Cf * Av * np.sqrt(n) / 3600
    ho = hi
    mi = mo = m

    exp.ports["in"].m.set_value(mi)
    exp.ports["out"].h.set_value(ho)
    exp.ports["out"].m.set_value(mo)


def rev_model(rev):

    p_in = rev.ports["in"].p.value
    m_in = rev.ports["in"].m.value
    fluid = rev.ports["in"].fluid
    p_out = p_in
    h_out = PropsSI('H', 'P', p_in, 'Q', 0.0, fluid)
    m_out = m_in
    rev.ports["out"].p.set_value(p_out)
    rev.ports["out"].h.set_value(h_out)
    rev.ports["out"].m.set_value(m_out)


def ihx_model(comp):

    k = 1000

    A = comp.parameter['A'].value

    hi_h = comp.ports["in_hot"].h.value - 1e-3
    pi_h = comp.ports["in_hot"].p.value
    mi_h = comp.ports["in_hot"].m.value
    hot_fluid = comp.ports["in_hot"].fluid

    hi_c = comp.ports["in_cold"].h.value + 1e-3
    pi_c = comp.ports["in_cold"].p.value
    mi_c = comp.ports["in_cold"].m.value
    cold_fluid = comp.ports["in_cold"].fluid

    C_h = mi_h * PropsSI('C', 'P', pi_h, 'H', hi_h, hot_fluid)
    C_c = mi_c * PropsSI('C', 'P', pi_c, 'H', hi_c, cold_fluid)
    T_in_h = PropsSI('T', 'P', pi_h, 'H', hi_h, hot_fluid)
    T_in_c = PropsSI('T', 'P', pi_c, 'H', hi_c, cold_fluid)

    R = C_h / C_c
    NTU = k * A / C_h
    n = (-1) * NTU * (1 + R)
    epsilon = (1 - np.exp(n)) / (1 + R)
    Q = epsilon * C_h * (T_in_h - T_in_c)

    po_c = pi_c
    po_h = pi_h
    ho_h = (hi_h - Q / mi_h)
    ho_c = (hi_c + Q / mi_c)
    mo_h = mi_h
    mo_c = mi_c

    comp.ports["out_hot"].p.set_value(po_h)
    comp.ports["out_hot"].h.set_value(ho_h)
    comp.ports["out_hot"].m.set_value(mo_h)

    comp.ports["out_cold"].p.set_value(po_c)
    comp.ports["out_cold"].h.set_value(ho_c)
    comp.ports["out_cold"].m.set_value(mo_c)


def subcooling_fun(net):
    h_s = PropsSI('H', 'P', net.components['Cond'].ports["out_r"].p.value,
                  'Q', 0.0, net.components['Cond'].ports["out_r"].fluid)
    res = (net.components['Cond'].ports["out_r"].h.value - h_s)
    return res


def superheat_fun(net):
    DC_value = 5.0
    T_sat = PropsSI('T', 'P', net.components['Evap'].ports["out_r"].p.value,
                    'Q', 1.0, net.components['Evap'].ports["out_r"].fluid)
    h_SH = PropsSI('H', 'T', T_sat + DC_value,
                   'P', net.components['Evap'].ports["out_r"].p.value,
                   net.components['Evap'].ports["out_r"].fluid)
    res = np.abs(net.components['Evap'].ports["out_r"].h.value - h_SH)
    return res


def cooling_load(network):
    Qsp = 18000
    Q = network.components['Evap'].outputs['Q'].value
    res = np.abs(Q - Qsp)
    return res


net = Network()

net.fluid_loops[1] = "R134a"
net.fluid_loops[2] = "INCOMP::AN[0.4]"
net.fluid_loops[3] = "INCOMP::AN[0.4]"

evap = MassFlowBasedComponent(
    label="Evap",
    port_specs={
        "in_af":  PortSpec("in"),
        "out_af": PortSpec("out"),
        "in_r": PortSpec("in"),
        "out_r": PortSpec("out"),
    },
)

comp = PressureBasedComponent(
    label="Comp",
    port_specs={
        "in":  PortSpec("in"),
        "out": PortSpec("out"),
    },
)

cond = MassFlowBasedComponent(
    label="Cond",
    port_specs={
        "in_r":  PortSpec("in"),
        "out_r": PortSpec("out"),
        "in_af": PortSpec("in"),
        "out_af": PortSpec("out"),
    },
)

exp = PressureBasedComponent(
    label="Exp",
    port_specs={
        "in":  PortSpec("in"),
        "out": PortSpec("out"),
    },
)

rev = MassFlowBasedComponent(
    label="Rev",
    port_specs={
        "in":  PortSpec("in"),
        "out": PortSpec("out"),
    },
)

ihx = MassFlowBasedComponent(
    label="IHX",
    port_specs={
        "in_hot":  PortSpec("in"),
        "out_hot": PortSpec("out"),
        "in_cold": PortSpec("in"),
        "out_cold": PortSpec("out"),
    },
)

net.add_component(comp)
net.add_component(cond)
net.add_component(evap)
net.add_component(exp)
net.add_component(ihx)
net.add_component(rev)

net.set_component_model("Comp", comp_model)
net.set_component_model("Cond", cond_model)
net.set_component_model("Evap", evap_model)
net.set_component_model("Exp", exp_model)
net.set_component_model("IHX", ihx_model)
net.set_component_model("Rev", rev_model)

net.connect("IHX", "out_cold", "Comp", "in",  fluid_loop=1)
net.connect("Comp", "out", "Cond", "in_r",   fluid_loop=1)
net.connect("Cond", "out_r", "Rev", "in", fluid_loop=1)
net.connect("Rev", "out", "IHX", "in_hot", fluid_loop=1)
net.connect("IHX", "out_hot", "Exp", "in", fluid_loop=1)
net.connect("Exp", "out", "Evap", "in_r", fluid_loop=1)
net.connect("Evap", "out_r", "IHX", "in_cold", fluid_loop=1)

net.add_parameter("Comp", "RPM")
net.add_parameter("Cond", "A")
net.add_parameter("Evap", "A")
net.add_parameter("IHX", "A")
net.add_parameter("Exp", "Cf")
net.add_parameter("Exp", "Av")

net.add_output(comp_label='Evap', output_label='Q')

net.add_constraint(label='subcool_eq', fun=subcooling_fun, ctype='eq', scale_factor=1e-5)
net.add_constraint(label='supheat_eq', fun=superheat_fun, ctype='obj', weight_factor=1e-5)
net.add_constraint(label='cool_load_eq', fun=cooling_load, ctype='obj', weight_factor=1e-3)

net.add_loop_breaker(fluid_loop=1, junction_id=1)

net.initialize()

# Boundary conditions at the evaporator antifrogen inlet and outlet ports
hi_evap_af = PropsSI("H", "P", 1e5, "T", -5.0+273.15, net.fluid_loops[2])
net.set_bc("Evap", "in_af", bc_type="input", var_type= "p", value=1e5, fluid_loop=2)
net.set_bc("Evap", "in_af", bc_type="input", var_type="h", value=hi_evap_af, fluid_loop=2)
net.set_bc("Evap", "in_af", bc_type="input", var_type="m", value=1.0, fluid_loop=2)
net.set_bc("Evap", "out_af", bc_type="output", var_type= "p", fluid_loop=2)
net.set_bc("Evap", "out_af", bc_type="output", var_type="h", fluid_loop=2)
net.set_bc("Evap", "out_af", bc_type="output", var_type="m", fluid_loop=2)

# Boundary conditions at the condenser antifrogen inlet and outlet ports
hi_cond_af = PropsSI("H", "P", 1e5, "T", 16.0+273.15, net.fluid_loops[3])
net.set_bc("Cond", "in_af", bc_type="input", var_type="p", value=1e5, fluid_loop=3)
net.set_bc("Cond", "in_af", bc_type="input", var_type="h", value=hi_cond_af, fluid_loop=3)
net.set_bc("Cond", "in_af", bc_type="input", var_type="m", value=1.0, fluid_loop=3)
net.set_bc("Cond", "out_af", bc_type="output", var_type="p", fluid_loop=3)
net.set_bc("Cond", "out_af", bc_type="output", var_type="h", fluid_loop=3)
net.set_bc("Cond", "out_af", bc_type="output", var_type="m", fluid_loop=3)

net.set_parameter("Comp", "RPM", value=2500, is_var=True, scale_factor=1.0e-3, bounds=(750, 3500))
net.set_parameter("Exp", "Av", value=0.3, is_var=True, scale_factor=1.0, bounds=(0.1, 1.0))
net.set_parameter("Exp", "Cf", value=0.25)
net.set_parameter("Cond", "A", value=3.65)
net.set_parameter("Evap", "A", value=2.56)
net.set_parameter("IHX", "A", value=0.75)

# net.print_tearing_variables()
x_init = [1.6e5, 4.1e5, 7.9e5, 7.9e5, 2.2e5, 1.6e5]
# x_init = [2.0e5, 4.0e5, 10.0e5, 10.0e5, 2.5e5, 2.0e5]

x_bnds = [(1.0e5, 4.0e5), (2.0e5, 6.0e5), (5.0e5, 30.0e5), (5.0e5, 30.0e5), (1.0e5, 3.5e5), (1.0e5, 4.0e5)]
net.set_inital_values(x_init, x_bnds)
net.print_residual_equations()

def main():
    net.solve_system()


if __name__ == "__main__":
    freeze_support()
    net.solve_system()
else:
    pass
def format_parameters(tree):

    # Read in the etree if a filename string is given
    if isinstance(tree, str):
        tree = etree.parse(tree)
    # Else, extract an etree if a tapper.experiment is given        
    elif isinstance(tree, Experiment):
        tree = tree.get_tree()
    else:
        pass

    opts = {}

    root = tree.getroot()
    # Only one (first) experiment can be handled for now 
    ex = root.findall('experiment')[0]
    
    if ex.find('geometry') is None or len(ex.find('geometry')) == 0:
        base = root
    else:
        base = ex

    zs = base.find('geometry').findall('zone')
    # assert that there are only three zones (implement one zone soon)
    assert(len(zs)==3)

    opts['length'], opts['porosity'], opts['D_ref'] = [], [], []

    opts['cross_section'] = float(zs[0].attrib['cross_section'])
    
    for i, z in enumerate(zs):
        opts['length'].append(float(z.attrib['length']))
        
        if len(z.findall('transport')) == 0:
            if base.find('geometry').find('transport') is not None:
                trans = base.find('geometry').find('transport')
            elif root.find('transport') is not None:
                trans = root.find('transport')
            else:
                raise Exception('No transport information for zone: %s' %
                                (tree.getpath(z),))
        else:
            try:
                trans = z.find('transport')
            except Exception as err:
                raise err('No transport information for zone: %s' %
                          (tree.getpath(z),))

        opts['porosity'].append(float(trans.attrib['porosity']))
        opts['D_ref'].append(float(trans.attrib['D_ref']))

    # assert that the middle zone has a material for which
    # pore diffusion kinetics is defined in the gloabl chemistry element
    # For now, pore diffusion kinetics in other zones and/or
    # other kinetics in any zone will be ignored.
    # For now, the first available catalyst from the second zone is used
    cat_name = zs[1].find('catalyst').attrib['name']
    mcat = zs[1].find('catalyst').attrib['mass']
    cat = root.find('chemistry').find('.//catalyst[@name="%s"]' % (cat_name))
    assert(len(cat.find('micropore'))>0)
    mpore = cat.find('micropore')
    
    opts['micropore_BC_type'] = mpore.find('BC').attrib['{http://www.w3.org/2001/XMLSchema-instance}type']
    if opts['micropore_BC_type']=='kinetics':
        opts['micropore_Ns'] = float(mpore.find('BC').attrib['Ns'])
        ka = float(mpore.find('BC').attrib['ka'])
        kd = float(mpore.find('BC').attrib['kd'])
    elif opts['micropore_BC_type']=='equilibrium':
        opts['micropore_Sv'] = float(mpore.find('BC').attrib['Sv'])
        KH = float(mpore.find('BC').attrib['KH'])

    opts['micropore_geometry'] = mpore.find('diffusion').attrib['geometry']
    opts['micropore_Rp'] = float(mpore.find('diffusion').attrib['Rp'])
    # For now, an isothermal value is expected.
    # In the future, to be extended to Arrhenius parameters
    Dp = float(mpore.find('diffusion').attrib['Dp'])
    
    if opts['micropore_BC_type']=='equilibrium':
        pars = zeros(2)
        pars[0] = Dp
        pars[1] = KH
    else:
        pars = zeros(3)
        pars[0] = Dp
        pars[1] = ka
        pars[2] = kd

    # handle one gas at a time, which means it must be marked in the tree
    gas = root.find('chemistry').find('.//gas[@micropore="true"]')
    mass = sum([int(x.attrib['n']) * float(el.attrib['mass'])
            for el in root.find('chemistry').findall('element')
            for x in gas.findall('element')
            if el.attrib['name'] == x.attrib['name']])
    opts['M'] = mass
    opts['gas'] = gas.attrib['name']
    
    ref_gas = root.find('chemistry').find('.//gas[@reference="true"]')
    ref_mass = sum([int(x.attrib['n']) * float(el.attrib['mass'])
                    for el in root.find('chemistry').findall('element')
                    for x in ref_gas.findall('element')
                    if el.attrib['name'] == x.attrib['name']])

    opts['Mref'] = ref_mass

    # inputSequence
    # For now, the first valve with declared Np is used and
    # pulsing from more than one valve cannot be accomodated.
    if ex.find('inputSequence') is None:
        base = root
    else:
        base = ex

    inSeq = base.find('inputSequence')
    valve = inSeq.find('.//valve[@Np]')
    Np_total = float(valve.attrib['Np'])
    fraction = float(valve.find('.//gas[@name="%s"]' % (gas.attrib['name'])).attrib['fraction'])
    opts['Np'] = Np_total*fraction
    opts['pTime'] = float(valve.attrib['pTime'])

    # outputSequence
    if ex.find('outputSequence') is None:
        base = root
    else:
        base = ex

    outSeq = base.find('outputSequence')
    opts['cTime'] = float(outSeq.attrib['cTime'])
    # For now, nSamples is equal to nPoints, which should be extended in the future
    opts['nSamples'] = float(outSeq.attrib['nPoints'])
    opts['cf'] = float(outSeq.find('.//response/gas[@name="%s"]' % (gas.attrib['name'])).attrib['cf'])

    # temperature
    opts['T'] = float(ex.find('.//field[@name="temperature"]').text)
    
    # general options
    opts['internal_BC_type'] = root.find('.//option[@key="internal_BC_type"]').attrib['value']
    opts['Tref'] = float(root.find('.//option[@key="T_ref"]').attrib['value'])
    opts['dx'] = 1.0/float(root.find('.//option[@key="integration"]').attrib['n_gridpoints_per_m'])
    opts['dr'] = 1.0/float(root.find('.//option[@key="micropore"]').attrib['n_gridpoints_per_m'])
    
    return pars, opts
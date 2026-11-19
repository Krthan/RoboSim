import xml.etree.ElementTree as ET
import numpy as np

from RoboticManipulator import Rotation, Transform, Joint, Link

def rpy_to_rot(rpy):
    r, p, y = rpy
    Rx = Rotation.RotationX(r)
    Ry = Rotation.RotationY(p)
    Rz = Rotation.RotationZ(y)
    
    return Rz @ Ry @ Rx

def getJointsDesc(urdf_path: str) -> dict:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    joints_desc = {}
    
    for item in root.findall('joint'):
        name = item.attrib["name"]
        j_type = item.attrib["type"]
        if j_type not in {"revolute", "continuous", "prismatic", "fixed"}:
            raise ValueError(f"Unsupported joint type '{j_type}' in joint '{name}'")
        parent = item.find("parent").attrib["link"]
        child = item.find("child").attrib["link"]
        if parent in joints_desc:
            raise ValueError(f"Link '{parent}' has more than one child joint; only serial chains are supported")

        # Axis (defaults to [1, 0, 0] for moving joints if omitted)
        axis_el = item.find("axis")
        if j_type != "fixed":
            axis_str = axis_el.attrib.get("xyz", "1 0 0") if axis_el is not None else "1 0 0"
            axis = np.array(axis_str.split(), dtype=float)
            axis = axis / np.linalg.norm(axis)
        else:
            axis = None
            
        # Origin
        origin_el = item.find("origin")
        if origin_el is not None:
            xyz = np.fromstring(origin_el.attrib.get("xyz", "0 0 0"), dtype=float, sep=" ")
            rpy = np.fromstring(origin_el.attrib.get("rpy", "0 0 0"), dtype=float, sep=" ")
        else:
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            
        # Limits
        limit_el = item.find("limit")
        if limit_el is not None:
            lower_limit = float(limit_el.attrib.get("lower", 0.0))
            upper_limit = float(limit_el.attrib.get("upper", 0.0))
        else:
            lower_limit, upper_limit = None, None
            
        joints_desc[parent] = {"name": name,
                                   "joint_type": j_type,
                                   "child": child,
                                   "xyz": xyz,
                                   "rpy": rpy,
                                   "axis": axis,
                                   "lower_limit": lower_limit,
                                   "upper_limit": upper_limit}
    
    return joints_desc

def getLinksDesc(urdf_path: str) -> dict:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    links_desc = {}
    
    for item in root.findall("link"):
        name = item.attrib["name"]
        inertial_el = item.find("inertial")
        
        if inertial_el is not None:
            # Center of Mass (CoM) origin relative to the link frame
            origin_el = inertial_el.find("origin")
            if origin_el is not None:
                com_xyz = np.fromstring(origin_el.attrib.get("xyz", "0 0 0"), dtype=float, sep=" ")
                com_rpy = np.fromstring(origin_el.attrib.get("rpy", "0 0 0"), dtype=float, sep=" ")
            else:
                com_xyz = np.zeros(3)
                com_rpy = np.zeros(3)
                
            # Mass
            mass_el = inertial_el.find("mass")
            mass = float(mass_el.attrib["value"]) if mass_el is not None else 0.0
            
            # 3x3 Symmetric Inertia Tensor
            inertia_el = inertial_el.find("inertia")
            if inertia_el is not None:
                ixx = float(inertia_el.attrib.get("ixx", 0.0))
                ixy = float(inertia_el.attrib.get("ixy", 0.0))
                ixz = float(inertia_el.attrib.get("ixz", 0.0))
                iyy = float(inertia_el.attrib.get("iyy", 0.0))
                iyz = float(inertia_el.attrib.get("iyz", 0.0))
                izz = float(inertia_el.attrib.get("izz", 0.0))
                
                inertia_matrix = np.array([
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz]
                ], dtype=float)
            else:
                inertia_matrix = np.zeros((3, 3))
        else:
            # Default values for massless / virtual links
            com_xyz = np.zeros(3)
            com_rpy = np.zeros(3)
            mass = 0.0
            inertia_matrix = np.zeros((3, 3))
            
        links_desc[name] = {"name": name, "mass": mass, "com_xyz": com_xyz, "com_rpy": com_rpy, "inertia": inertia_matrix}
        
    return links_desc
    

def getJointsAndLinksFromURDF(urdf_path: str) -> tuple[list[Joint], list[Link]]:
    joints_desc = getJointsDesc(urdf_path=urdf_path)
    links_desc = getLinksDesc(urdf_path=urdf_path)
    
    T_world_current = Transform(np.zeros((3, 1)), np.eye(3)) # base link
    current_link = getBaseLink(urdf_path) # in the start parent_link is the base link
    joint_pivots_world = []
    joint_axes_world = []
    joint_types = []
    p_com_home = []
    masses = []
    inertias = []
    parent_link_list = []

    while current_link in joints_desc:
        jd = joints_desc[current_link]
        child_link = jd["child"]

        T_current_child = Transform(jd["xyz"], rpy_to_rot(jd["rpy"]))
        T_world_child = T_world_current @ T_current_child # child link's frame in world

        # Child link's CoM transform in world frame
        l_data = links_desc[child_link]
        T_com_child = Transform(l_data["com_xyz"], rpy_to_rot(l_data["com_rpy"]))
        T_com_world_child = T_world_child @ T_com_child

        # Rotate inertia tensor into Base-aligned CoM frame: I_base = R * I_local * R^T
        R_com = T_com_world_child.R
        I_aligned = R_com @ l_data["inertia"] @ R_com.T
        p_com = T_com_world_child.p
        m = l_data["mass"]

        if jd["joint_type"] != "fixed":
            # Joint pivot and axis in world frame
            joint_pivots_world.append(T_world_child.p)
            joint_axes_world.append(T_world_child.R @ jd["axis"])
            joint_types.append(jd["joint_type"])
            p_com_home.append(p_com)
            masses.append(m)
            inertias.append(I_aligned)
            parent_link_list.append(current_link)
        elif masses and m > 0.0:
            # Fixed joint: merge this link's mass into the last moving link
            # (combined CoM, inertia shifted with the parallel axis theorem).
            m0, c0, I0 = masses[-1], p_com_home[-1], inertias[-1]
            m_tot = m0 + m
            c_tot = (m0 * c0 + m * p_com) / m_tot
            I_tot = np.zeros((3, 3))
            for mk, ck, Ik in ((m0, c0, I0), (m, p_com, I_aligned)):
                d = (ck - c_tot).ravel()
                I_tot += Ik + mk * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
            masses[-1], p_com_home[-1], inertias[-1] = m_tot, c_tot, I_tot

        T_world_current = T_world_child
        current_link = child_link
    
    joints = []
    links = []
    
    for i in range(len(joint_axes_world)):
        p_prev = np.zeros((3, 1)) if i == 0 else p_com_home[i-1]
        M_rel_home = Transform(p = (p_com_home[i] - p_prev),R = np.eye(3))
        
        w_i = joint_axes_world[i]
        if joint_types[i] == "prismatic":
            S_i = np.concatenate([np.zeros(3), w_i])
        else:
            r_pivot = joint_pivots_world[i] - p_com_home[i]
            v_i = -np.cross(w_i.ravel(), r_pivot.ravel())
            S_i = np.concatenate([w_i, v_i])
        
        joints.append(Joint(f"Joint_{i+1}", S_i, parent=parent_link_list[i], child=joints_desc[parent_link_list[i]]["child"], 
                            limits = {k: joints_desc[parent_link_list[i]][k] for k in ("lower_limit", "upper_limit")}))
        links.append(Link(f"Link_{i+1}", masses[i], inertias[i], M_rel_home))
    
    return joints, links

def getBaseLink(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    all_links = {l.attrib["name"] for l in root.findall("link")}
    child_links = {j.find("child").attrib["link"] for j in root.findall("joint")}

    base_link = (all_links - child_links).pop()
    
    return base_link

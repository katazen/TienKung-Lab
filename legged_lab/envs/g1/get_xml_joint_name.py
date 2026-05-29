import xml.etree.ElementTree as ET


def get_robot_details(xml_file):
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 1. 获取所有关节 (Joint) 的名字顺序
    # 查找所有名为 joint 的标签
    joints = [j.get('name') for j in root.findall('.//joint') if j.get('name')]

    # 2. 获取执行器 (Actuator) 下关联的关节顺序
    actuator_joint_names = []

    # 找到 <actuator> 节点
    actuator_root = root.find('.//actuator')

    if actuator_root is not None:
        # 遍历 actuator 下的所有子节点 (可能是 motor, position, velocity 等)
        for act in actuator_root:
            # Mujoco 格式中，关节名通常直接写在 joint 属性里
            joint_name = act.get('joint')
            if joint_name:
                actuator_joint_names.append(joint_name)
            else:
                # 如果执行器本身有名字但没写 joint 属性，也可以记录它自己的名字
                act_name = act.get('name')
                if act_name:
                    actuator_joint_names.append(f"actuator_name: {act_name}")

    return joints, actuator_joint_names

if __name__ == '__main__':

    # --- 运行测试 ---
    xml_path = '/home/woan/workspace/TienKung-Lab/legged_lab/assets/g1/g1_23dof_rev_1_0.xml'
    try:
        joint_list, actuator_list = get_robot_details(xml_path)

        print("【1. 关节顺序 (Joints)】")
        print(joint_list)

        print("\n【2. 执行器关联顺序 (Actuators)】")
        print(actuator_list)

    except Exception as e:
        print(f"解析失败: {e}")
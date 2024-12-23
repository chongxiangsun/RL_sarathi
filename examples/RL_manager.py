import logging
import numpy as np
import tensorflow as tf
class RLManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RLManager, cls).__new__(cls)
            cls._instance.actor = None
            cls._instance.critic = None
        return cls._instance

    @staticmethod
    def initialize(state_dim, action_dim, actor_lr, critic_lr,action_dim_list):
        """初始化 Actor 和 Critic"""
        instance = RLManager()
        if instance.actor is None:
            instance.actor = RLManager.Actor(state_dim=state_dim, action_dim=action_dim, action_dim_list=action_dim_list, lr=actor_lr)
        if instance.critic is None:
            instance.critic = RLManager.Critic(state_dim=state_dim, lr=critic_lr)
        logging.info("Actor and Critic initialized successfully.")

    @staticmethod
    def get_actor():
        """获取 Actor 实例"""
        instance = RLManager()
        if instance.actor is None:
            raise ValueError("Actor has not been initialized. Call RLManager.initialize() first.")
        return instance.actor

    @staticmethod
    def get_critic():
        """获取 Critic 实例"""
        instance = RLManager()
        if instance.critic is None:
            raise ValueError("Critic has not been initialized. Call RLManager.initialize() first.")
        return instance.critic

    class Actor:
        def __init__(self, state_dim, action_dim, action_dim_list,lr,epsilon=0.1):
            self.action_dim = action_dim  # 动作空间维度
            self.action_dim_list = action_dim_list  # 每个动作维度的取值范围列表
            self.old_policy = RLManager.build_actor_network(state_dim,  action_dim_list)  # 创建旧策略网络
            self.new_policy = RLManager.build_actor_network(state_dim,  action_dim_list)  # 创建新策略网络
            self.update_policy()  # 初始化时将旧策略更新为新策略
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # 创建 Adam 优化器用于训练新策略网络
            self.epsilon = epsilon  # 添加 epsilon 参数用于ε-贪婪策略
        
        # 根据当前状态选择动作
        def choice_action(self, in_state):
            # 使用旧策略网络计算当前状态下的动作概率分布，停止梯度以确保在训练时不会更新这个网络
            # print(f"Input State: {in_state}")

            local_policy = tf.stop_gradient(self.old_policy(
                np.array([in_state])
            )).numpy()[0]  # 获取概率分布的 NumPy 数组并选择第一个（也是唯一的）元素
            # 从动作概率分布中随机选择一个动作
            return np.random.choice(
                np.prod(self.action_dim_list),# 动作空间的维度
                p=local_policy  # 动作的概率分布
            ), local_policy  # 返回选择的动作和策略概率分布



        # 将新策略网络的权重赋值给旧策略网络
        def update_policy(self):
            self.old_policy.set_weights(
                self.new_policy.get_weights()  # 将新策略网络的权重复制到旧策略网络
            )

        # 更新新策略网络
        def learn(self, batch_state, batch_action, le_advantage, epsilon=0.2, entropy_weight=0.01):
            le_advantage = np.reshape(le_advantage, newshape=(-1))  # 将优势函数数组重塑为一维数组
            # 创建动作索引，用于从策略网络中提取对应的动作概率
            batch_action = tf.stack([tf.range(tf.shape(batch_action)[0], dtype=tf.int32), batch_action], axis=1)
            old_policy = self.old_policy(batch_state)  # 计算旧策略网络下的动作概率
            with tf.GradientTape() as tape:
                new_policy = self.new_policy(batch_state)  # 计算新策略网络下的动作概率
                # 从新旧策略中提取对应动作的概率
                pi_prob = tf.gather_nd(params=new_policy, indices=batch_action)
                old_policy_prob = tf.gather_nd(params=old_policy, indices=batch_action)
                # print(f"pi_prob: {pi_prob.numpy()}")
                # print(f"old_policy_prob: {old_policy_prob.numpy()}")

                ratio = pi_prob / (old_policy_prob + 1e-6)  # 计算概率比（重要度采样比）
                surr1 = ratio * le_advantage  # 计算目标函数的第一个部分
                surr2 = tf.clip_by_value(ratio, clip_value_min=1.0 - epsilon, clip_value_max=1.0 + epsilon) * le_advantage
                # 计算目标函数的第二部分，进行裁剪以稳定训练
                ppo_loss = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO 损失

                # 熵奖励
                entropy = -tf.reduce_sum(new_policy * tf.math.log(new_policy + 1e-6), axis=1)
                entropy_bonus = tf.reduce_mean(entropy)  # 平均熵

                # 总损失：PPO 损失 - 熵奖励
                loss = ppo_loss - entropy_weight * entropy_bonus  
                # loss = - tf.reduce_mean(tf.minimum(surr1, surr2))  # 计算损失函数，取两个部分中的较小值的均值并取负值（因为我们最小化损失）
            grad = tape.gradient(loss, self.new_policy.trainable_variables)  # 计算损失函数对新策略网络权重的梯度
            # print(f"Gradient: {grad}")
            # print(f"Loss: {loss.numpy()}")
            clipped_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in grad]  # 对梯度进行裁剪
            self.optimizer.apply_gradients(zip(clipped_grad, self.new_policy.trainable_variables))

    class Critic:
        def __init__(self, state_dim, lr):
            self.value = RLManager.build_critic_network(state_dim)  # 创建价值网络
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # 创建 Adam 优化器用于训练价值网络

        def get_advantage(self, get_state, get_reward):
            """计算优势函数"""
            return get_reward - self.value.predict(get_state, verbose=0)  # 当前奖励减去预测的状态价值

        def get_value(self, input_state):
            """计算状态的价值"""
            return self.value.predict(input_state, verbose=0)  # 预测给定状态的价值

        def learn(self, batch_state, batch_reward):
            """更新 Critic 的价值网络"""
            with tf.GradientTape() as tape:
                value_predict = self.value(batch_state)  # 计算价值网络的预测
                loss = tf.keras.losses.mean_squared_error(batch_reward, value_predict)  # 计算均方误差损失
            grad = tape.gradient(loss, self.value.trainable_variables)  # 计算梯度
            # clipped_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in grad]  # 对梯度进行裁剪
            self.optimizer.apply_gradients(zip(grad, self.value.trainable_variables))


    @staticmethod
    def build_actor_network(state_dim, action_dim_list):
        # 计算总的输出维度长度
        total_action_dim = np.prod(action_dim_list)  # 动作空间的总维度
        # print(f"Total action dim: {total_action_dim}")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=state_dim, activation='tanh'),  # 第一层，状态空间维度
            tf.keras.layers.Dense(units=512, activation='tanh'),  # 第二层，隐藏层，512个节点
            tf.keras.layers.Dense(units=total_action_dim, activation='softmax')  # 输出层，动作空间维度
            
        ])
        model.build(input_shape=(None, state_dim))  # 定义输入形状
        return model

    @staticmethod
    def build_critic_network(state_dim):
        """构建 Critic 网络"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=state_dim, activation='tanh'),  # 第一层，状态空间维度
            tf.keras.layers.Dense(units=512, activation='tanh'),  # 第二层，隐藏层，512个节点
            tf.keras.layers.Dense(units=1, activation='linear')  # 输出层，单值表示状态价值
        ])
        model.build(input_shape=(None, state_dim))
        return model


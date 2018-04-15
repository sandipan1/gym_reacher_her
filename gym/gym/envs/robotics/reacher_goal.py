import numpy as np
from gym import utils
from gym.envs.robotics import rotations, robot_env, utils

def goal_distance(goal_a, goal_b):
	assert goal_a.shape == goal_b.shape
	return np.linalg.norm(goal_a - goal_b, axis=-1)


class ReacherGoal(robot_env.RobotEnv):
	def __init__(self, model_path,reward_type,distance_threshold,n_substeps=10):
		self.distance_threshold=distance_threshold
		self.reward_type=reward_type
		#self.num_action=2
		self.initial_qpos=np.array([0.0,0.0,0.0,0.0])

		super(ReacherGoal, self).__init__(
			model_path=model_path, n_substeps=n_substeps, n_actions=2,
			initial_qpos=self.initial_qpos)

	def compute_reward(self, achieved_goal, goal, info):

		# Compute distance between goal and the achieved goal.
		d = goal_distance(achieved_goal,goal)
		if self.reward_type == 'sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			return -d


	def _step_callback(self):
		'''can be called after each step of simulation'''
		

	def _set_action(self, action):
		assert action.shape == (2,)
		action = np.array(action.copy())  # ensure that we don't change the action outside of this scope
		action *= 1  
		action = action.clip(-0.5,0.5)
		# Apply action to simulation.
		utils.ctrl_set_action(self.sim, action)
		#utils.mocap_set_action(self.sim, action)
		for i in range(60):
			self.sim.step()
	def _get_obs(self):
		# positions
		finger_xpos =self.sim.data.get_body_xpos("robot:fingertip")
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep
		finger_xvelp = self.sim.data.get_body_xvelp('robot:fingertip') * dt
		a=self.sim.model.joint_names
		valqpos=np.array(list(map(self.sim.data.get_joint_qpos,a)))
		valqvel=np.array(list(map(self.sim.data.get_joint_qvel,a)))
		valqvel=valqvel * dt
		obj_pos=self.sim.data.get_site_xpos("target")
		obj_velp = self.sim.data.get_site_xvelp('target') * dt
		obj_velr = self.sim.data.get_site_xvelr('target') * dt
		obj_rel_pos= obj_pos - finger_xpos

		obs=np.concatenate([finger_xpos,obj_pos,obj_rel_pos,valqpos[:2],finger_xvelp,obj_velp,obj_velr,valqvel[:2]])


		return {
			'observation': obs.copy(),
			'achieved_goal': finger_xpos[:2].copy(),
			'desired_goal': self.goal.copy(),
		}
	def _render_callback(self):
		# Visualize target.
		#sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
		pass
	def _viewer_setup(self):
		
		self.viewer.cam.trackbodyid = 0

   
	def _reset_sim(self):
		for i in range(2):
			self.sim.data.ctrl[i] = self.initial_qpos[i]
		for i in range(60):
			self.sim.step()	
		return True
# radius =0.2
	def _sample_goal(self):
		while (1):
			qpos = self.initial_state.qpos[-2:] + .21*np.array([self.np_random.uniform(low=0, high=1),self.np_random.uniform(low=-1, high=1)])
			if np.linalg.norm(qpos) <= .2 and np.linalg.norm(qpos) >= 0.03:
				break
		site_id = self.sim.model.site_name2id('target')
		self.sim.model.site_pos[site_id][:2] = qpos
		self.sim.model.site_pos[site_id][-1] = 0.01

		self.sim.forward()
		for i in range(60):
			self.sim.step()
		return qpos



	def _is_success(self, achieved_goal, desired_goal):
		d = goal_distance(achieved_goal, desired_goal)
		return (d < self.distance_threshold).astype(np.float32)

	def _env_setup(self, initial_qpos):
		pass




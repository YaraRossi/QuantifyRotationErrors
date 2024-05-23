import numpy
from numpy import sin, cos, tan, arccos, pi
from tqdm import tqdm

def rot_vec(phi, theta, psi, data):
    """
        Applies a rotation matrix to a given vector based on Euler angles.

        Parameters:
        phi (float): The roll angle (rotation about the x-axis) in radians.
        theta (float): The pitch angle (rotation about the y-axis) in radians.
        psi (float): The yaw angle (rotation about the z-axis) in radians.
        data (numpy.ndarray): The input vector to be rotated. Shape: (3,).

        Returns:
        numpy.ndarray: The rotated vector after applying the rotation matrix. Shape: (3,).

        Notes:
        - The rotation matrix R is computed based on the given Euler angles (phi, theta, psi).
        - The function returns the result of applying the rotation matrix R to the input vector data.
        """
    R = numpy.array([[cos(theta) * cos(psi), cos(theta) * sin(psi), -sin(theta)],
                     [sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
                      sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi), cos(theta) * sin(phi)],
                     [cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
                      cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi), cos(theta) * cos(phi)]])

    return R.dot(data)

def matrix_AE(phi, theta):
    """
        Computes the rotation matrix for body to local coordinate system transformation.

        Parameters:
        phi (float): The roll angle (rotation about the x-axis) in radians.
        theta (float): The pitch angle (rotation about the y-axis) in radians.

        Returns:
        numpy.ndarray: A 3x3 rotation matrix representing the transformation from body to local
        system using Euler angles.

        Notes:
        - The rotation matrix AE is computed based on the given x and y angles (phi and theta).
        - sec(x) = 1/cos(x)
        - The function returns a 3x3 numpy array representing the rotation matrix AE.
        """
    # sec(x) = 1/cos(x)
    AE = numpy.array([[1, sin(phi) * tan(theta), cos(phi)*tan(theta)],
                      [0, cos(phi), -sin(phi)],
                      [0, sin(phi) * 1/cos(theta), cos(phi) * 1/cos(theta)]])
    return AE

# not sure if this one is correct!!!
'''def attitude_equation_trapez(dt=0, obs_rate = '', obs = '', earth_rr = ''):
    length = len(obs_rate[0,:])
    euler = numpy.zeros((3,length))
    for i in tqdm(range(1, length)):
        euler[i] = euler[i-1] + dt/2 * (matrix_AE(phi=euler[0, i-1], theta=euler[0, i-1]).dot(obs_rate[:,i-1]+earth_rr)
                           + matrix_AE(phi=obs[0, i], theta=obs[1, i]).dot(obs_rate[:, i]+earth_rr)
                           - 2 * earth_rr)
    return euler'''

# here it is assumed that the rotational sensor is not demeaned but shows the rotation rate of the earth.
def attitude_equation_simple(dt=0, obs_rate = '', earth_rr = ''):
    """
    Computes attitude angles and rates with and without correction for Earth's rotation.

    Parameters:
    dt (float): Time step for integration. Default is 0.
    obs_rate (numpy.ndarray): Array of observed rotation rates. Shape: (3, length).
    earth_rr (numpy.ndarray): Array of Earth's rotation rates. Shape: (3, 1).

    Returns:
    tuple: A tuple containing the following arrays:
        euler_a (numpy.ndarray): Euler angles without Earth rotation correction. Shape: (3, length).
        euler_a_err (numpy.ndarray): Euler angles with Earth rotation correction. Shape: (3, length).
        euler_rr (numpy.ndarray): Euler rotation rates without Earth rotation correction. Shape: (3, length).
        euler_rr_err (numpy.ndarray): Euler rotation rates with Earth rotation correction. Shape: (3, length).

    Notes:
    - It is assumed that the rotational sensor is not demeaned but shows the rotation rate of the Earth.
    - The function integrates Euler angle and rate equations with and without correction for Earth's rotation.
    - The rotation rate of the Earth is best taken directly from the rotational sensor itself, rather than
    calculated from the latitude.
    """

    # Initialize arrays to store calculated angles and rotation rates
    length = len(obs_rate[0, :])
    euler_a = numpy.zeros((3, length))
    rot_a_err = numpy.zeros((3, length))
    euler_a_tot = numpy.zeros((3, length))
    euler_rr = numpy.zeros((3, length))
    rot_rr_err = numpy.zeros((3, length))
    euler_rr_tot = numpy.zeros((3, length))

    for i in tqdm(range(1, length)):
        ###############
        # Euler Angles without correction for Earth's rotation
        phi, theta = euler_a[0, i-1], euler_a[1, i-1] # Get previous Euler angles
        # Earth rotation rate is subtracted before applying the attitude correction for the signal.
        euler_a[:, i] = euler_a[:, i-1] + dt * \
                     matrix_AE(phi=phi, theta=theta).dot(obs_rate[:, i-1] - earth_rr)

        ###############
        # Rotation Angles with correction for Earth rotation
        phi, theta, psi = rot_a_err[0, i-1], rot_a_err[1, i-1], rot_a_err[2, i-1]
        rot_a_err[:, i] = rot_a_err[:, i-1] + dt * \
                     (obs_rate[:, i-1] - rot_vec(phi=phi, theta=theta, psi=psi, data=earth_rr))

        ###############
        # Euler Angles with correction for Earth's rotation
        phi, theta, psi = euler_a_tot[0, i-1], euler_a_tot[1, i-1], euler_a_tot[2, i-1] # Get previous Euler angles
        # Earth rotation rate is first corrected for the dynamic rotation before subtracting from signal
        euler_a_tot[:, i] = euler_a_tot[:, i - 1] + dt * \
                            (matrix_AE(phi=phi, theta=theta).dot(obs_rate[:, i - 1]) -
                             rot_vec(phi=phi, theta=theta, psi=psi, data=earth_rr))
        ###############
        # Euler rates without correction for Earth's rotation
        phi, theta = euler_a[0, i], euler_a[1, i] # Get nows Euler angles
        # Earth rotation rate is subtracted before applying the attitude correction for the signal.
        euler_rr[:, i] = matrix_AE(phi=phi, theta=theta).dot(obs_rate[:, i] - earth_rr)

        # Rot rates with correction for Earth rotation
        phi, theta, psi = rot_a_err[0, i], rot_a_err[1, i], rot_a_err[2, i]
        rot_rr_err[:, i] = obs_rate[:, i] - rot_vec(phi=phi, theta=theta, psi=psi, data=earth_rr)

        ###############
        # Euler rates with correction for Earth's rotation
        phi, theta, psi = euler_a_tot[0, i], euler_a_tot[1, i], euler_a_tot[2, i]# Get nows Euler angles
        # Earth rotation rate is first corrected for the dynamic rotation before subtracting from signal
        euler_rr_tot[:, i] = matrix_AE(phi=phi, theta=theta).dot(obs_rate[:, i]) - \
                             rot_vec(phi=phi, theta=theta, psi=psi, data=earth_rr)

    # Return computed arrays
    # euler_a: euler angles without earth rotation correction
    # euler_a_err: euler angles with earth rotation correction
    # euler_rr: euler rotation rate without earth rotation correction
    # euler_rr_err: euler rotation rate with earth rotation correction
    return euler_a, rot_a_err, euler_rr, rot_rr_err, euler_a_tot, euler_rr_tot

def earth_rotationrate(Latitude):
    """
        Calculates the Earth's rotation rate at a given latitude.

        Parameters:
        Latitude (float): The latitude of the location in degrees.

        Returns:
        numpy.ndarray: An array containing the Earth's rotation rates in different directions

        Notes:
        - The East-West rotation rate is always zero.
        """
    EarthRR = (365.25 + 1) * (2 * pi) / (365.25 * 24 * 3600)
    VertiRR = sin(Latitude * pi / 180) * EarthRR  # (rad/s) in Up direction (down if the value is negative)
    NorthRR = cos(Latitude * pi / 180) * EarthRR  # (rad/s) in North direction (as EarthRR is always North)
    earth_obs = numpy.array([0, NorthRR, VertiRR])
    return earth_obs
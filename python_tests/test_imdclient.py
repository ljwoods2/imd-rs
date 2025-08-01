"""Test for IMDClient functionality"""

import logging

import pytest
from numpy.testing import (
    assert_allclose,
)
import MDAnalysis as mda
from MDAnalysisTests.datafiles import (
    COORDINATES_TOPOLOGY,
    COORDINATES_H5MD,
)
from quickstream import IMDClient
import imdclient
from imdclient.IMDClient import imdframe_memsize
from imdclient.IMDProtocol import IMDHeaderType
from imdclient.tests.utils import (
    create_default_imdsinfo_v3,
)
from imdclient.tests.server import InThreadIMDServer


# logger = logging.getLogger("imdclient.IMDClient")
# file_handler = logging.FileHandler("test.log")
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# logger.setLevel(logging.DEBUG)

IMDENERGYKEYS = [
    "step",
    "temperature",
    "total_energy",
    "potential_energy",
    "van_der_walls_energy",
    "coulomb_energy",
    "bonds_energy",
    "angles_energy",
    "dihedrals_energy",
    "improper_dihedrals_energy",
]


class TestIMDClientV3:

    @pytest.fixture
    def universe(self):
        return mda.Universe(COORDINATES_TOPOLOGY, COORDINATES_H5MD)

    @pytest.fixture
    def imdsinfo(self):
        return create_default_imdsinfo_v3()

    @pytest.fixture
    def server_client_two_frame_buf(self, universe, imdsinfo):
        server = InThreadIMDServer(universe.trajectory)
        server.set_imdsessioninfo(imdsinfo)
        server.handshake_sequence("localhost", first_frame=False)
        client = IMDClient(
            f"localhost",
            server.port,
            universe.trajectory.n_atoms,
            buffer_size=imdframe_memsize(universe.trajectory.n_atoms, imdsinfo) * 2,
        )
        server.join_accept_thread()
        yield server, client
        client.stop()
        server.cleanup()

    @pytest.fixture(params=[">", "<"])
    def server_client(self, universe, imdsinfo, request):
        server = InThreadIMDServer(universe.trajectory)
        imdsinfo.endianness = request.param
        server.set_imdsessioninfo(imdsinfo)
        server.handshake_sequence("localhost", first_frame=False)
        client = IMDClient(
            f"localhost",
            server.port,
            universe.atoms.n_atoms,
        )
        server.join_accept_thread()
        yield server, client
        client.stop()
        server.cleanup()

    @pytest.fixture
    def server_client_incorrect_atoms(self, universe, imdsinfo):
        server = InThreadIMDServer(universe.trajectory)
        server.set_imdsessioninfo(imdsinfo)
        server.handshake_sequence("localhost", first_frame=False)
        client = IMDClient(
            f"localhost",
            server.port,
            universe.atoms.n_atoms + 1,
        )
        server.join_accept_thread()
        yield server, client
        client.stop()
        server.cleanup()

    def test_traj_unchanged(self, server_client, universe):
        server, client = server_client
        server.send_frames(0, 5)
        for i in range(5):
            imdf = client.get_imdframe()
            assert_allclose(universe.trajectory[i].time, imdf.time)
            assert_allclose(universe.trajectory[i].dt, imdf.dt)
            assert_allclose(universe.trajectory[i].data["step"], imdf.step)
            assert_allclose(universe.trajectory[i].positions, imdf.positions)
            assert_allclose(universe.trajectory[i].velocities, imdf.velocities)
            assert_allclose(universe.trajectory[i].forces, imdf.forces)
            assert_allclose(universe.trajectory[i].triclinic_dimensions, imdf.box)

    def test_pause_resume_continue(self, server_client_two_frame_buf):
        server, client = server_client_two_frame_buf
        server.send_frames(0, 2)
        # Client's buffer is filled. client should send pause
        server.expect_packet(IMDHeaderType.IMD_PAUSE)
        # Empty buffer
        client.get_imdframe()
        # only the second call actually frees buffer memory
        client.get_imdframe()
        # client has free memory. should send resume
        server.expect_packet(IMDHeaderType.IMD_RESUME)
        server.send_frame(1)
        client.get_imdframe()

    def test_pause_resume_disconnect(self, server_client_two_frame_buf):
        """Client pauses because buffer is full, empties buffer and attempt to resume, but
        finds that simulation has already ended and raises EOF"""
        server, client = server_client_two_frame_buf
        server.send_frames(0, 2)
        server.expect_packet(IMDHeaderType.IMD_PAUSE)
        client.get_imdframe()
        client.get_imdframe()
        # client has free frame. should send resume
        server.expect_packet(IMDHeaderType.IMD_RESUME)
        # simulation is over. client should raise EOF
        server.disconnect()
        with pytest.raises(EOFError):
            client.get_imdframe()

    def test_pause_resume_no_disconnect(self, server_client_two_frame_buf):
        """Client pauses because buffer is full, empties buffer and attempt to resume, but
        finds that simulation has already ended (but has not yet disconnected) and raises EOF
        """
        server, client = server_client_two_frame_buf
        server.send_frames(0, 2)
        server.expect_packet(IMDHeaderType.IMD_PAUSE)
        client.get_imdframe()
        client.get_imdframe()
        # client has free frame. should send resume
        server.expect_packet(IMDHeaderType.IMD_RESUME)
        # simulation is over. client should raise EOF
        with pytest.raises(EOFError):
            client.get_imdframe()
        # server should receive disconnect from client (though it doesn't have to do anything)
        server.expect_packet(IMDHeaderType.IMD_DISCONNECT)

    @pytest.mark.parametrize("cont", [True, False])
    def test_continue_after_disconnect(self, universe, imdsinfo, cont):
        server = InThreadIMDServer(universe.trajectory)
        server.set_imdsessioninfo(imdsinfo)
        server.handshake_sequence("localhost", first_frame=False)
        client = IMDClient(
            f"localhost",
            server.port,
            universe.trajectory.n_atoms,
            continue_after_disconnect=cont,
        )
        server.join_accept_thread()
        server.expect_packet(IMDHeaderType.IMD_WAIT, expected_length=(int)(not cont))

    def test_incorrect_atom_count(self, server_client_incorrect_atoms, universe):
        server, client = server_client_incorrect_atoms

        server.send_frame(0)

        with pytest.raises(EOFError) as exc_info:
            client.get_imdframe()

        error_msg = str(exc_info.value)
        assert f"Expected n_atoms value {universe.atoms.n_atoms + 1}" in error_msg
        assert f"got {universe.atoms.n_atoms}" in error_msg
        assert "Ensure you are using the correct topology file" in error_msg


class TestIMDClientV3ContextManager:
    @pytest.fixture
    def universe(self):
        return mda.Universe(COORDINATES_TOPOLOGY, COORDINATES_H5MD)

    @pytest.fixture
    def imdsinfo(self):
        return create_default_imdsinfo_v3()

    @pytest.fixture
    def server(self, universe, imdsinfo):
        server = InThreadIMDServer(universe.trajectory)
        server.set_imdsessioninfo(imdsinfo)
        yield server
        server.cleanup()

    def test_context_manager_traj_unchanged(self, server, universe):
        server.handshake_sequence("localhost", first_frame=False)

        i = 0
        with IMDClient(
            "localhost",
            server.port,
            universe.trajectory.n_atoms,
        ) as client:
            server.send_frames(0, 5)
            while i < 5:

                imdf = client.get_imdframe()
                assert_allclose(universe.trajectory[i].time, imdf.time)
                assert_allclose(universe.trajectory[i].dt, imdf.dt)
                assert_allclose(universe.trajectory[i].data["step"], imdf.step)
                assert_allclose(universe.trajectory[i].positions, imdf.positions)
                assert_allclose(universe.trajectory[i].velocities, imdf.velocities)
                assert_allclose(universe.trajectory[i].forces, imdf.forces)
                assert_allclose(universe.trajectory[i].triclinic_dimensions, imdf.box)
                i += 1
        server.expect_packet(IMDHeaderType.IMD_DISCONNECT)
        assert i == 5

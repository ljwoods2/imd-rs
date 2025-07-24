use byteorder::{BigEndian, ByteOrder, ReadBytesExt, WriteBytesExt};
use crossbeam_channel::{unbounded, Receiver, Sender};
use log::{debug, error, info, trace, warn};
use ndarray::Array2;
use pyo3::prelude::*;
use std::{
    io::{self, ErrorKind, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    rc::Rc,
    thread::{self, JoinHandle},
    time::Duration,
    vec,
};

#[repr(i32)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum IMDMessageType {
    Disconnect = 0,
    Energies = 1,
    FCoords = 2,
    Go = 3,
    Handshake = 4,
    Kill = 5,
    MDComm = 6,
    Pause = 7,
    TRate = 8,
    IOError = 9,
    SessionInfo = 10,
    Resume = 11,
    Time = 12,
    Box = 13,
    Velocities = 14,
    Forces = 15,
    Wait = 16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub enum Endianness {
    Big,
    Little,
}

pub type IMDHeaderType = IMDMessageType;

impl std::convert::TryFrom<i32> for IMDMessageType {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(IMDMessageType::Disconnect),
            1 => Ok(IMDMessageType::Energies),
            2 => Ok(IMDMessageType::FCoords),
            3 => Ok(IMDMessageType::Go),
            4 => Ok(IMDMessageType::Handshake),
            5 => Ok(IMDMessageType::Kill),
            6 => Ok(IMDMessageType::MDComm),
            7 => Ok(IMDMessageType::Pause),
            8 => Ok(IMDMessageType::TRate),
            9 => Ok(IMDMessageType::IOError),
            10 => Ok(IMDMessageType::SessionInfo),
            11 => Ok(IMDMessageType::Resume),
            12 => Ok(IMDMessageType::Time),
            13 => Ok(IMDMessageType::Box),
            14 => Ok(IMDMessageType::Velocities),
            15 => Ok(IMDMessageType::Forces),
            16 => Ok(IMDMessageType::Wait),
            _ => Err("Invalid value for IMDMessageType"),
        }
    }
}

const IMDHEADERSIZE: usize = 8;
const IMDVERSIONS: &[i32] = &[2, 3];

#[derive(Debug, Clone, Copy)]
struct IMDHeader {
    header_type: IMDMessageType,
    length: i32,
}

impl IMDHeader {
    pub fn write_to(&self, writer: &mut impl WriteBytesExt) -> io::Result<()> {
        writer.write_i32::<BigEndian>(self.header_type as i32)?;
        writer.write_i32::<BigEndian>(self.length)?;
        Ok(())
    }

    pub fn from_reader(reader: &mut impl ReadBytesExt) -> io::Result<Self> {
        println!("Reading header type val");
        let header_type_val = reader.read_i32::<BigEndian>()?;
        println!("Reading length");
        let length = reader.read_i32::<BigEndian>()?;

        let header_type = IMDMessageType::try_from(header_type_val)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;

        Ok(IMDHeader {
            header_type,
            length,
        })
    }

    fn new(header_type: IMDMessageType, length: i32) -> Self {
        IMDHeader {
            header_type: header_type,
            length: length,
        }
    }

    pub fn go() -> Self {
        Self::new(IMDMessageType::Go, 0)
    }
    pub fn handshake(version: i32) -> Self {
        Self::new(IMDMessageType::Handshake, version)
    }
    pub fn session_info() -> Self {
        Self::new(IMDMessageType::SessionInfo, 7)
    }
    pub fn disconnect() -> Self {
        Self::new(IMDMessageType::Disconnect, 0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IMDSessionInfo {
    pub version: i32,
    pub endianness: Endianness,
    pub wrapped_coords: bool,
    pub time: bool,
    pub energies: bool,
    pub box_info: bool,
    pub positions: bool,
    pub velocities: bool,
    pub forces: bool,
}

impl IMDSessionInfo {
    fn from_reader(
        conn: &mut impl ReadBytesExt,
        endianness: Endianness,
    ) -> io::Result<IMDSessionInfo> {
        let time = conn.read_u8()? != 0;
        let energies = conn.read_u8()? != 0;
        let box_info = conn.read_u8()? != 0;
        let positions = conn.read_u8()? != 0;
        let wrapped_coords = conn.read_u8()? != 0;
        let velocities = conn.read_u8()? != 0;
        let forces = conn.read_u8()? != 0;

        Ok(IMDSessionInfo {
            version: 3,
            endianness: endianness,
            wrapped_coords: wrapped_coords,
            time: time,
            energies: energies,
            box_info: box_info,
            positions: positions,
            velocities: velocities,
            forces: forces,
        })
    }

    fn frame_size_bytes(&self, n_atoms: u64) -> u64 {
        let mut memsize = 0;

        if self.time {
            memsize += 8 * 3;
        }
        if self.energies {
            memsize += 4 * 10;
        }
        if self.box_info {
            memsize += 4 * 9;
        }
        if self.positions {
            memsize += 4 * 3 * n_atoms;
        }

        if self.velocities {
            memsize += 4 * 3 * n_atoms
        }
        if self.forces {
            memsize += 4 * 3 * n_atoms
        }

        memsize
    }
}
#[derive(Debug, Clone, Copy)]
pub struct IMDEnergies {
    pub step: i32,
    pub temperature: f32,
    pub total: f32,
    pub potential: f32,
    pub vdw: f32,
    pub coulomb: f32,
    pub bonds: f32,
    pub angles: f32,
    pub dihedrals: f32,
    pub impropers: f32,
}
#[derive(Debug, Clone, Copy)]
pub struct IMDTime {
    pub step: i32,
    pub dt: f64,
    pub time: f64,
}
#[derive(Debug, Clone)]
pub struct IMDFrame {
    pub time: Option<IMDTime>,
    pub positions: Option<Array2<f32>>,
    pub velocities: Option<Array2<f32>>,
    pub forces: Option<Array2<f32>>,
    pub energies: Option<IMDEnergies>,
    pub box_info: Option<Array2<f32>>,
}

impl IMDFrame {
    pub fn new(n_atoms: u64, sinfo: IMDSessionInfo) -> Self {
        let n = n_atoms as usize;

        IMDFrame {
            time: if sinfo.time {
                Some(IMDTime {
                    step: 0,
                    dt: 0.0,
                    time: 0.0,
                })
            } else {
                None
            },
            positions: if sinfo.positions {
                Some(Array2::zeros((n, 3)))
            } else {
                None
            },
            velocities: if sinfo.velocities {
                Some(Array2::zeros((n, 3)))
            } else {
                None
            },
            forces: if sinfo.forces {
                Some(Array2::zeros((n, 3)))
            } else {
                None
            },
            energies: if sinfo.energies {
                Some(IMDEnergies {
                    step: 0,
                    temperature: 0.0,
                    total: 0.0,
                    potential: 0.0,
                    vdw: 0.0,
                    coulomb: 0.0,
                    bonds: 0.0,
                    angles: 0.0,
                    dihedrals: 0.0,
                    impropers: 0.0,
                })
            } else {
                None
            },
            box_info: if sinfo.box_info {
                Some(Array2::zeros((3, 3)))
            } else {
                None
            },
        }
    }

    // fn load_xvf_from(&mut self, R: &mut impl ReadBytesExt, buf: &mut Vec<u8>) {}
}

struct IMDProducerV2 {
    energy_buf: Option<Vec<u8>>,
    xvf_buf: Option<Vec<u8>>,
}

impl IMDProducerV2 {
    fn new(sinfo: IMDSessionInfo, n_atoms: u64) -> IMDProducerV2 {
        unimplemented!()
    }
    fn parse_imdframe(
        &self,
        reader: &mut impl ReadBytesExt,
        empty_frame: IMDFrame,
    ) -> io::Result<IMDFrame> {
        unimplemented!()
    }
}
struct IMDProducerV3 {
    time_buf: Option<Vec<u8>>,
    energy_buf: Option<Vec<u8>>,
    xvf_buf: Option<Vec<u8>>,
    box_buf: Option<Vec<u8>>,
}

impl IMDProducerV3 {
    fn new(sinfo: IMDSessionInfo, n_atoms: u64) -> IMDProducerV3 {
        let time_buf: Option<Vec<u8>> = match sinfo.time {
            true => Some(vec![0u8; 40]),
            false => None,
        };

        let xvf_buf: Option<Vec<u8>> = match sinfo.positions | sinfo.velocities | sinfo.forces {
            // 4 bytes * n atoms * 3
            true => Some(vec![0u8; 12 * n_atoms as usize]),
            false => None,
        };

        let box_buf: Option<Vec<u8>> = match sinfo.box_info {
            // 4 bytes * 9
            true => Some(vec![0u8; 36]),
            false => None,
        };

        let energy_buf: Option<Vec<u8>> = match sinfo.energies {
            // 4 bytes * 10
            true => Some(vec![0u8; 40]),
            false => None,
        };

        IMDProducerV3 {
            time_buf,
            energy_buf,
            xvf_buf,
            box_buf,
        }
    }

    fn parse_imdframe(
        &self,
        reader: &mut impl ReadBytesExt,
        empty_frame: IMDFrame,
    ) -> io::Result<IMDFrame> {
        unimplemented!()
    }
}

enum IMDProducerType {
    V2(IMDProducerV2),
    V3(IMDProducerV3),
}

struct IMDProducer {
    conn: TcpStream,
    empty_recv: Receiver<IMDFrame>,
    full_send: Sender<IMDFrame>,
    sinfo: IMDSessionInfo,
    n_atoms: u64,
    version_handler: IMDProducerType,
    paused: bool,
}

impl IMDProducer {
    pub fn new(
        conn: TcpStream,
        empty_recv: Receiver<IMDFrame>,
        full_send: Sender<IMDFrame>,
        sinfo: IMDSessionInfo,
        n_atoms: u64,
    ) -> io::Result<Self> {
        let version_handler = match sinfo.version {
            2 => IMDProducerType::V2(IMDProducerV2::new(sinfo, n_atoms)),
            3 => IMDProducerType::V3(IMDProducerV3::new(sinfo, n_atoms)),
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!("Unsupported IMD version: {}", sinfo.version),
                ))
            }
        };

        Ok(IMDProducer {
            conn,
            empty_recv,
            full_send,
            sinfo,
            n_atoms,
            version_handler,
            paused: false,
        })
    }

    pub fn run(&mut self) -> io::Result<()> {
        loop {
            let empty_frame = match self.empty_recv.recv() {
                Ok(frame) => frame,
                Err(e) => {
                    return Err(io::Error::new(
                        ErrorKind::ConnectionAborted,
                        format!("Channel recv failed: {e}"),
                    ));
                }
            };

            let full_frame = match &mut self.version_handler {
                IMDProducerType::V2(v2) => v2.parse_imdframe(&mut self.conn, empty_frame)?,
                IMDProducerType::V3(v3) => v3.parse_imdframe(&mut self.conn, empty_frame)?,
            };

            if let Err(e) = self.full_send.send(full_frame) {
                return Err(io::Error::new(
                    ErrorKind::ConnectionAborted,
                    format!("Channel send failed: {e}"),
                ));
            }
        }
    }
}

pub struct IMDClient {
    sinfo: IMDSessionInfo,
    producer_handle: thread::JoinHandle<io::Result<()>>,
    full_recv: Receiver<IMDFrame>,
    empty_send: Sender<IMDFrame>,
    prev_frame: Option<IMDFrame>,
}

impl IMDClient {
    pub fn new<A: ToSocketAddrs>(
        addr: A,
        n_atoms: u64,
        buffer_size: Option<u64>,
    ) -> io::Result<Self> {
        println!("Connecting!");
        let mut conn = TcpStream::connect(&addr)?;

        println!("Connected, waiting for handshake!");

        let sinfo = Self::await_imd_handshake(&mut conn)?;

        let max_bytes = match buffer_size {
            Some(size) => size,
            None => 10 * 1024u64.pow(2),
        };
        let frame_bytes = sinfo.frame_size_bytes(n_atoms);
        let num_frames = max_bytes / frame_bytes;

        let (full_send, full_recv) = unbounded();
        let (empty_send, empty_recv) = unbounded();

        for _ in 0..num_frames {
            empty_send.send(IMDFrame::new(n_atoms, sinfo));
        }

        let producer_conn = conn.try_clone()?;

        let producer_handle =
            Self::start_producer_thread(producer_conn, empty_recv, full_send, sinfo, n_atoms);

        IMDHeader::go().write_to(&mut conn)?;

        Ok(IMDClient {
            sinfo,
            producer_handle,
            full_recv,
            empty_send,
            prev_frame: None,
        })
    }

    pub fn get_imdframe(&mut self) -> io::Result<&mut IMDFrame> {
        let new_imdframe = self
            .full_recv
            .recv()
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;

        let prev_frame = self.prev_frame.take();

        if prev_frame.is_none() {
            self.empty_send.send(prev_frame.expect("Error"));
        }

        self.prev_frame = Some(new_imdframe);

        self.prev_frame
            .as_mut()
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "Failed"))
    }

    pub fn get_imdsessioninfo(&mut self) -> io::Result<IMDSessionInfo> {
        Ok(self.sinfo)
    }

    pub fn stop(&mut self) {
        unimplemented!()
    }

    fn await_imd_handshake(conn: &mut TcpStream) -> io::Result<IMDSessionInfo> {
        let mut version: Option<i32> = None;
        let mut endianness: Option<Endianness> = None;

        conn.set_read_timeout(Some(Duration::from_secs(5)))?;

        // self.conn.read_i32();
        // self.conn.read_exact(&mut h_buf).map_err(|e| {
        //     io::Error::new(
        //         e.kind(),
        //         format!("IMDClient: No handshake packet received: {}", e),
        //     )
        // })?;

        let header = IMDHeader::from_reader(conn)?;

        print!("Got header!");

        println!("{:?}", header);

        if header.header_type != IMDMessageType::Handshake {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Expected header type `IMD_HANDSHAKE`, got {:?}",
                    header.header_type
                ),
            ));
        }

        if !IMDVERSIONS.contains(&header.length) {
            // Try swapping endianness
            let swapped = header.length.swap_bytes();
            if !IMDVERSIONS.contains(&swapped) {
                let err_version = std::cmp::min(swapped, header.length);
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Incompatible IMD version. Expected version in {:?}, got {}",
                        IMDVERSIONS, err_version
                    ),
                ));
            } else {
                endianness = Some(Endianness::Big);
                version = Some(swapped);
            }
        } else {
            endianness = Some(Endianness::Big);
            version = Some(header.length);
        }

        let endianness = endianness.unwrap();
        let version = version.unwrap();

        if version == 2 {
            // IMD v2 does not send a configuration handshake body packet
            return Ok(IMDSessionInfo {
                version,
                endianness: endianness,
                wrapped_coords: false,
                time: false,
                energies: true,
                box_info: false,
                positions: true,
                velocities: false,
                forces: false,
            });
        } else if version == 3 {
            let header = IMDHeader::from_reader(conn)?;

            if header.header_type != IMDMessageType::SessionInfo {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Expected header type `IMD_SESSIONINFO`, got {:?}",
                        header.header_type
                    ),
                ));
            }

            if header.length != 7 {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!("Expected header length 7, got {}", header.length),
                ));
            }

            return IMDSessionInfo::from_reader(conn, endianness);
        } else {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Unsupported IMD version: {}", version),
            ));
        };
    }

    fn start_producer_thread(
        conn: TcpStream,
        empty_recv: Receiver<IMDFrame>,
        full_send: Sender<IMDFrame>,
        sinfo: IMDSessionInfo,
        n_atoms: u64,
    ) -> thread::JoinHandle<Result<(), io::Error>> {
        thread::spawn(move || -> Result<(), io::Error> {
            let mut producer = IMDProducer::new(conn, empty_recv, full_send, sinfo, n_atoms)?;
            producer.run()
        })
    }
}

pub struct IMDServer {
    listener: TcpListener,
    conn: Option<TcpStream>,
    port: u16,
}

impl IMDServer {
    pub fn new<A: ToSocketAddrs>(addr: A, blocking: bool) -> io::Result<Self> {
        let listener = TcpListener::bind(addr).unwrap();
        listener.set_nonblocking(!blocking);
        let port = listener.local_addr().unwrap().port();

        Ok(IMDServer {
            listener,
            conn: None,
            port,
        })
    }

    pub fn try_accept(&mut self) -> io::Result<Option<SocketAddr>> {
        match self.listener.accept() {
            Ok((stream, addr)) => {
                self.conn = Some(stream);
                Ok(Some(addr))
            }
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }

    // pub fn handshake(&mut self, ) {}
}

#[cfg(feature = "python")]
mod python_bindings {
    use super::IMDEnergies;
    use super::IMDTime;

    use super::IMDClient as RustIMDClient;
    use super::IMDFrame as RustIMDFrame;
    use super::IMDSessionInfo as RustIMDSessionInfo;
    use ndarray::ArrayView2;
    use numpy::PyArray2;
    use numpy::ToPyArray;
    use pyo3::ffi::getter;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use pyo3::Bound;

    #[pyclass]
    pub struct IMDClient {
        inner: RustIMDClient,
    }

    // #[pyclass]
    // pub struct IMDFrame {
    //     positions: Py<PyArray2<f32>>,
    // }

    // #[pymethods]
    // impl IMDFrame {
    //     #[getter]
    //     fn positions<'py>(
    //         this: Bound<'py, Self>,
    //         py: Python<'py>,
    //     ) -> PyResult<Bound<'py, PyArray2<f32>>> {
    //         let array = this.borrow().positions.as_ref(py);
    //         Ok(array.bind(py))
    //     }
    // }

    // #[pymethods]
    // impl IMDFrame {
    //     #[getter]
    //     fn positions<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyArray2<f32>>> {
    //         match &self.positions_view {
    //             Some(view) => Ok(Some(view.to_pyarray(py))),
    //             None => Ok(None),
    //         }
    //     }

    //     #[getter]
    //     fn velocities<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyArray2<f32>>> {
    //         match &self.velocities_view {
    //             Some(view) => Ok(Some(view.to_pyarray(py))),
    //             None => Ok(None),
    //         }
    //     }

    //     #[getter]
    //     fn forces<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyArray2<f32>>> {
    //         match &self.forces_view {
    //             Some(view) => Ok(Some(view.to_pyarray(py))),
    //             None => Ok(None),
    //         }
    //     }

    //     #[getter]
    //     fn box_info<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyArray2<f32>>> {
    //         match &self.box_info_view {
    //             Some(view) => Ok(Some(view.to_pyarray(py))),
    //             None => Ok(None),
    //         }
    //     }

    //     #[getter]
    //     fn time(&self) -> Option<(i32, f64, f64)> {
    //         self.time
    //     }

    //     #[getter]
    //     fn energies(&self) -> Option<(i32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> {
    //         self.energies
    //     }
    // }

    #[pyclass]
    pub struct IMDSessionInfo {
        inner: RustIMDSessionInfo,
    }

    #[pymethods]
    impl IMDSessionInfo {
        #[getter]
        fn version(&self) -> i32 {
            self.inner.version
        }

        #[getter]
        fn endianness(&self) -> &str {
            match self.inner.endianness {
                super::Endianness::Big => ">",
                super::Endianness::Little => "<",
            }
        }

        #[getter]
        fn wrapped_coords(&self) -> bool {
            self.inner.wrapped_coords
        }

        #[getter]
        fn time(&self) -> bool {
            self.inner.time
        }

        #[getter]
        fn energies(&self) -> bool {
            self.inner.energies
        }

        #[getter]
        fn box_info(&self) -> bool {
            self.inner.box_info
        }

        #[getter]
        fn positions(&self) -> bool {
            self.inner.positions
        }

        #[getter]
        fn velocities(&self) -> bool {
            self.inner.velocities
        }

        #[getter]
        fn forces(&self) -> bool {
            self.inner.forces
        }
    }

    #[pymethods]
    impl IMDClient {
        #[new]
        #[pyo3(signature = (host, port, n_atoms, buffer_size=None, continue_after_disconnect=None))]
        fn new(
            py: Python,
            host: &str,
            port: u16,
            n_atoms: u64,
            buffer_size: Option<u64>,
            continue_after_disconnect: Option<bool>,
        ) -> PyResult<Self> {
            py.allow_threads(|| {
                let addr = format!("{}:{}", host, port);
                let client = RustIMDClient::new(addr, n_atoms, buffer_size)?;
                Ok(IMDClient { inner: client })
            })
        }
        fn get_imdframe<'py>(
            this: Bound<'py, Self>,
            py: Python<'py>,
        ) -> PyResult<Bound<'py, PyDict>> {
            let rust_client = &mut this.borrow_mut().inner;

            let rust_frame = py
                .allow_threads(|| rust_client.get_imdframe())
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyEOFError, _>(format!("IO error: {}", e))
                })?;

            let dict = pyo3::types::PyDict::new(py);

            let anchor = this.into_any();

            let positions = match &rust_frame.positions {
                Some(p) => Some(unsafe { PyArray2::borrow_from_array(p, anchor.clone()) }),
                None => None,
            };

            dict.set_item("positions", positions)?;

            let velocities = match &rust_frame.velocities {
                Some(p) => Some(unsafe { PyArray2::borrow_from_array(p, anchor.clone()) }),
                None => None,
            };

            dict.set_item("velocties", velocities)?;

            let forces = match &rust_frame.forces {
                Some(p) => Some(unsafe { PyArray2::borrow_from_array(p, anchor.clone()) }),
                None => None,
            };

            dict.set_item("forces", forces)?;

            Ok(dict)
        }

        fn get_imdsessioninfo<'py>(
            this: Bound<'py, Self>,
            py: Python<'py>,
        ) -> PyResult<Bound<'py, IMDSessionInfo>> {
            let rust_client = &mut this.borrow_mut().inner;

            let rust_sinfo = py
                .allow_threads(|| rust_client.get_imdsessioninfo())
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyEOFError, _>(format!("IO error: {}", e))
                })?;

            let sinfo = IMDSessionInfo { inner: rust_sinfo };

            Ok(Py::new(py, sinfo)?.into_bound(py))
        }
    }

    #[pymodule]
    fn quickstream(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<IMDClient>()?;
        // m.add_class::<IMDFrame>()?;
        m.add_class::<IMDSessionInfo>()?;
        Ok(())
    }
}
// #[derive(Debug)]
// struct IMDHeader {
//     header_type: IMDMessageType,
//     length: i32,
// }

// fn expect_imd_handshake(stream: &mut TcpStream) -> () {
//     let mut buf = [0; 8];

//     match stream.read_exact(&mut buf) {
//         Ok(()) => {
//             let handshake_header_int = i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);

//             let length = i32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
//         }
//         Err(e) => {
//             eprintln!("Error reading from stream: {}", e);
//         }
//     }
//     1;
// }

// fn expect_imd_header(
//     stream: &mut TcpStream,
//     imd_type: IMDMessageType,
//     buf: &mut [u8; 8],
//     val: Option<i32>,
// ) -> io::Result<IMDHeader> {
//     match stream.read_exact(buf) {
//         Ok(()) => {
//             let header_type = i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);

//             let length = i32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
//         }
//         Err(e) => {
//             eprintln!("Error reading from stream: {}", e);
//         }
//     }

//     if let Some(expected_val) = val {
//         {}
//     }

//     Ok()
// }

#[cfg(test)]
mod tests {

    use super::*;

    // #[test]
    // fn server_client() {
    //     let mut imdserver = IMDServer::new("127.0.0.1:0", true).unwrap();
    //     let port = imdserver.port;

    //     let handle = thread::spawn(move || {
    //         let client_addr = imdserver.try_accept();
    //         imdserver
    //             .conn
    //             .as_mut()
    //             .unwrap()
    //             .write_i32::<BigEndian>(IMDMessageType::Handshake as i32)
    //             .unwrap();

    //         imdserver
    //             .conn
    //             .as_mut()
    //             .unwrap()
    //             .write_i32::<BigEndian>(3)
    //             .unwrap();
    //     });

    //     let address = format!("127.0.0.1:{}", port);
    //     let mut imdclient = IMDClient::new(address, 64);
    // }

    // #[test]
    // fn server_and_client_connect() {
    //     let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind listener");

    //     let addr = listener.local_addr().unwrap();
    //     let ip = addr.ip();
    //     let port = addr.port();

    //     // 2) Spawn the server thread that will accept one connection
    //     let handle = thread::spawn(move || {
    //         // blocks until a client connects
    //         let (mut server_stream, client_addr) =
    //             listener.accept().expect("failed to accept connection");

    //         assert_eq!(client_addr.ip().to_string(), "127.0.0.1");

    //         // Echo one message back to client
    //         let mut buf = [0u8; 5];
    //         server_stream.read_exact(&mut buf).unwrap();
    //         assert_eq!(&buf, b"hello");
    //         server_stream.write_all(b"world").unwrap();
    //     });

    //     let mut imdclient = IMDClient::new(addr, 64);
    //     // assert_eq!(client_stream.peer_addr().unwrap(), addr);

    //     // // Send a small message and read the echo
    //     // client_stream.write_all(b"hello").unwrap();
    //     // let mut resp = [0u8; 5];
    //     // client_stream.read_exact(&mut resp).unwrap();
    //     // assert_eq!(&resp, b"world");

    //     // 4) Wait for the server thread to finish
    //     // handle.join().expect("server thread panicked");
    // }
}

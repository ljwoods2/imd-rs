use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use crossbeam_channel::{unbounded, Receiver, Sender};
use log::{debug, error, info, trace, warn};
use ndarray::Array2;
use pyo3::prelude::*;
use std::{
    io::{self, BufRead, BufReader, ErrorKind, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    rc::Rc,
    sync::{Arc, Condvar, Mutex},
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

    fn from_reader(reader: &mut impl ReadBytesExt) -> io::Result<Self> {
        let header_type_val = reader.read_i32::<BigEndian>()?;

        let length = reader.read_i32::<BigEndian>()?;

        let header_type = IMDMessageType::try_from(header_type_val)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;

        Ok(IMDHeader {
            header_type,
            length,
        })
    }

    fn refill_from_reader(&mut self, reader: &mut impl ReadBytesExt) -> io::Result<()> {
        self.header_type = IMDMessageType::try_from(reader.read_i32::<BigEndian>()?)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;

        self.length = reader.read_i32::<BigEndian>()?;

        debug!("{:?}", self.header_type);
        debug!("{:?}", self.length);

        Ok(())
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
    pub fn pause() -> Self {
        Self::new(IMDMessageType::Pause, 0)
    }
    pub fn resume() -> Self {
        Self::new(IMDMessageType::Resume, 0)
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
    pub step: i64,
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

// enum IMDVersion {
//     V2,
//     V3,
// }

// struct IMDProducerV2 {
//     energy_buf: Option<Vec<u8>>,
//     xvf_buf: Option<Vec<u8>>,
// }

// impl IMDProducerV2 {
//     fn new(sinfo: IMDSessionInfo, n_atoms: u64) -> IMDProducerV2 {
//         unimplemented!()
//     }
//     fn parse_imdframe(
//         &self,
//         reader: &mut impl ReadBytesExt,
//         empty_frame: IMDFrame,
//     ) -> io::Result<IMDFrame> {
//         unimplemented!()
//     }
// }
// struct IMDProducerV3 {
//     time_buf: Option<Vec<u8>>,
//     energy_buf: Option<Vec<u8>>,
//     xvf_buf: Option<Vec<u8>>,
//     box_buf: Option<Vec<u8>>,
// }

// impl IMDProducerV3 {
//     fn new(sinfo: IMDSessionInfo, n_atoms: u64) -> IMDProducerV3 {
//         let time_buf: Option<Vec<u8>> = match sinfo.time {
//             true => Some(vec![0u8; 40]),
//             false => None,
//         };

//         let xvf_buf: Option<Vec<u8>> = match sinfo.positions | sinfo.velocities | sinfo.forces {
//             // 4 bytes * n atoms * 3
//             true => Some(vec![0u8; 12 * n_atoms as usize]),
//             false => None,
//         };

//         let box_buf: Option<Vec<u8>> = match sinfo.box_info {
//             // 4 bytes * 9
//             true => Some(vec![0u8; 36]),
//             false => None,
//         };

//         let energy_buf: Option<Vec<u8>> = match sinfo.energies {
//             // 4 bytes * 10
//             true => Some(vec![0u8; 40]),
//             false => None,
//         };

//         IMDProducerV3 {
//             time_buf,
//             energy_buf,
//             xvf_buf,
//             box_buf,
//         }
//     }

//     fn parse_imdframe(
//         &self,
//         reader: &mut impl ReadBytesExt,
//         empty_frame: IMDFrame,
//     ) -> io::Result<IMDFrame> {
//         unimplemented!()
//     }
// }

// enum IMDProducerType {
//     V2(IMDProducerV2),
//     V3(IMDProducerV3),
// }
#[derive(Debug)]
struct Shared {
    consumer_finished: bool,
}

struct IMDProducer {
    state: Arc<(Mutex<Shared>, Condvar)>,
    conn: TcpStream,
    reader: BufReader<TcpStream>,
    empty_recv: Receiver<IMDFrame>,
    full_send: Sender<IMDFrame>,
    num_frames: u64,
    error_send: Sender<io::Error>,
    sinfo: IMDSessionInfo,
    n_atoms: u64,
    paused: bool,
    header_buf: IMDHeader,
    // time_buf: Option<[u8; 24]>,
    // energy_buf: Option<[u8; 40]>,
    // xvf_buf: Option<Vec<u8>>,
    // box_buf: Option<[u8; 36]>,
}

impl IMDProducer {
    pub fn new(
        state: Arc<(Mutex<Shared>, Condvar)>,
        conn: TcpStream,
        empty_recv: Receiver<IMDFrame>,
        full_send: Sender<IMDFrame>,
        num_frames: u64,
        error_send: Sender<io::Error>,
        sinfo: IMDSessionInfo,
        n_atoms: u64,
    ) -> io::Result<Self> {
        // just fill it with placeholder values
        let header_buf = IMDHeader::new(IMDMessageType::IOError, 0);

        let time_buf: Option<[u8; 24]> = match sinfo.time {
            true => Some([0u8; 24]),
            false => None,
        };

        let xvf_buf: Option<Vec<u8>> = match sinfo.positions | sinfo.velocities | sinfo.forces {
            // 4 bytes * n atoms * 3
            true => Some(vec![0u8; 12 * n_atoms as usize]),
            false => None,
        };

        let box_buf: Option<[u8; 36]> = match sinfo.box_info {
            // 4 bytes * 9``
            true => Some([0u8; 36]),
            false => None,
        };
        let energy_buf: Option<[u8; 40]> = match sinfo.energies {
            // 4 bytes * 10
            true => Some([0u8; 40]),
            false => None,
        };

        let reader = BufReader::new(conn.try_clone()?);

        Ok(IMDProducer {
            state,
            conn,
            reader,
            empty_recv,
            full_send,
            num_frames,
            error_send,
            sinfo,
            n_atoms,
            paused: false,
            header_buf,
            // time_buf,
            // energy_buf,
            // xvf_buf,
            // box_buf,
        })
    }

    fn data_available(&mut self) -> io::Result<bool> {
        debug!("Checking if data is available");
        match self.reader.fill_buf() {
            Ok(buf) => Ok(!buf.is_empty()), // true if there's data
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(false), // no data yet
            Err(e) => Err(e),               // other errors bubble up
        }
    }

    fn wait_for_space(&mut self) -> io::Result<()> {
        debug!("Waiting for space");
        if self.empty_recv.len() as f64 > 0.5 * self.num_frames as f64 {
            return Ok(());
        } else {
            let (lock, cvar) = &*self.state;
            // acquire the lock
            debug!("Acquiring lock");
            let mut shared = lock
                .lock()
                .map_err(|_poisoned| io::Error::new(ErrorKind::InvalidData, "lock poisoned"))?;

            // loop exactly like your Python `while …: cond.wait()`
            while (self.empty_recv.len() as f64 > 0.5 * self.num_frames as f64
                && !shared.consumer_finished)
            {
                debug!("IMDProducer: Waiting…");
                debug!(
                    "IMDProducer: consumer_finished: {}",
                    shared.consumer_finished
                );

                // this atomically unlocks the mutex and sleeps;
                // when woken, it re-locks and returns the guard
                shared = cvar
                    .wait(shared)
                    .map_err(|_poisoned| io::Error::new(ErrorKind::InvalidData, "lock poisoned"))?;
            }
            Ok(())
        }
    }

    pub fn run(&mut self) -> io::Result<()> {
        debug!("Producer starting");
        loop {
            if !self.paused {
                // arbitrarily
                // if <25% of frames are empty
                // the consumer is working too slow for the simulation
                // so pause it
                debug!("Checking pause condition");
                if self.empty_recv.len() as f64 <= 0.25 * self.num_frames as f64 {
                    debug!("Pausing producer");
                    IMDHeader::pause().write_to(&mut self.conn)?;
                    debug!("Stn paused signal");
                    self.paused = true;
                    // to fail out when partial frames are parsed in a paused
                    // state, rather than wait forever
                    self.conn.set_nonblocking(true)?;
                    debug!("Set socket to nonblocking");
                }
            }
            debug!("Checking if unpause condition met");
            if self.paused && !self.data_available()? {
                debug!("unpause condition met");
                self.wait_for_space()?;
                debug!("Waited for space");
                IMDHeader::resume().write_to(&mut self.conn)?;
                self.paused = false;
                self.conn.set_nonblocking(false)?;
            }

            debug!("starting loop");
            let empty_frame = match self.empty_recv.recv() {
                Ok(frame) => frame,
                Err(e) => {
                    let _ = IMDHeader::disconnect().write_to(&mut self.conn);
                    let err_msg = format!("Channel recv failed: {e}");
                    let err = io::Error::new(ErrorKind::Other, err_msg.clone());
                    let _ = self
                        .error_send
                        .send(io::Error::new(ErrorKind::Other, err_msg));
                    return Err(err);
                }
            };

            debug!("Got empty frame");

            let full_frame = self.parse_frame(empty_frame);

            let full_frame = match full_frame {
                Ok(f) => f,
                Err(e) => {
                    let _ = IMDHeader::disconnect().write_to(&mut self.conn);
                    let err_msg = format!("{e}");
                    let _ = self
                        .error_send
                        .send(io::Error::new(ErrorKind::Other, err_msg.clone()));
                    return Err(io::Error::new(ErrorKind::Other, err_msg));
                }
            };

            debug!("Parsed a frame");

            if let Err(e) = self.full_send.send(full_frame) {
                let _ = IMDHeader::disconnect().write_to(&mut self.conn);
                let err_msg = format!("Channel send failed: {e}");
                let err = io::Error::new(ErrorKind::ConnectionAborted, err_msg.clone());
                let _ = self
                    .error_send
                    .send(io::Error::new(ErrorKind::Other, err_msg));
                return Err(err);
            }

            debug!("Pushed a frame");
        }
    }

    fn parse_frame(&mut self, empty_frame: IMDFrame) -> io::Result<IMDFrame> {
        match self.sinfo.version {
            2 => self.parse_imdframe_v2(empty_frame),
            3 => self.parse_imdframe_v3(empty_frame),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid version",
            )),
        }
    }

    fn parse_imdframe_v2(&mut self, empty_frame: IMDFrame) -> io::Result<IMDFrame> {
        unimplemented!()
    }

    fn parse_imdframe_v3(&mut self, empty_frame: IMDFrame) -> io::Result<IMDFrame> {
        let mut full_frame = empty_frame;

        debug!("Expecting time...");

        if self.sinfo.time {
            self.expect(IMDMessageType::Time, Some(1))?;
            let time = full_frame.time.as_mut().unwrap();

            match self.sinfo.endianness {
                Endianness::Big => {
                    time.dt = self.reader.read_f64::<BigEndian>()?;
                    time.time = self.reader.read_f64::<BigEndian>()?;
                    time.step = self.reader.read_i64::<BigEndian>()?;
                }
                Endianness::Little => {
                    time.dt = self.reader.read_f64::<LittleEndian>()?;
                    time.time = self.reader.read_f64::<LittleEndian>()?;
                    time.step = self.reader.read_i64::<LittleEndian>()?;
                }
            }

            debug!("Time packet: {:?}", time);
        }

        if self.sinfo.energies {
            debug!("Got time, expecting energies...");
            self.expect(IMDMessageType::Energies, Some(1))?;
            let energies = full_frame.energies.as_mut().unwrap();

            match self.sinfo.endianness {
                Endianness::Big => {
                    energies.step = self.reader.read_i32::<BigEndian>()?;
                    energies.temperature = self.reader.read_f32::<BigEndian>()?;
                    energies.total = self.reader.read_f32::<BigEndian>()?;
                    energies.potential = self.reader.read_f32::<BigEndian>()?;
                    energies.vdw = self.reader.read_f32::<BigEndian>()?;
                    energies.coulomb = self.reader.read_f32::<BigEndian>()?;
                    energies.bonds = self.reader.read_f32::<BigEndian>()?;
                    energies.angles = self.reader.read_f32::<BigEndian>()?;
                    energies.dihedrals = self.reader.read_f32::<BigEndian>()?;
                    energies.impropers = self.reader.read_f32::<BigEndian>()?;
                }
                Endianness::Little => {
                    energies.step = self.reader.read_i32::<LittleEndian>()?;
                    energies.temperature = self.reader.read_f32::<LittleEndian>()?;
                    energies.total = self.reader.read_f32::<LittleEndian>()?;
                    energies.potential = self.reader.read_f32::<LittleEndian>()?;
                    energies.vdw = self.reader.read_f32::<LittleEndian>()?;
                    energies.coulomb = self.reader.read_f32::<LittleEndian>()?;
                    energies.bonds = self.reader.read_f32::<LittleEndian>()?;
                    energies.angles = self.reader.read_f32::<LittleEndian>()?;
                    energies.dihedrals = self.reader.read_f32::<LittleEndian>()?;
                    energies.impropers = self.reader.read_f32::<LittleEndian>()?;
                }
            }
            debug!("NRG packet: {:?}", energies);
        }

        debug!("Got energies, expecting box..");

        if self.sinfo.box_info {
            self.expect(IMDMessageType::Box, Some(1))?;
            Self::read_into_array2(
                &mut self.reader,
                &mut full_frame.box_info.as_mut().unwrap(),
                self.sinfo.endianness,
            )?;
        }

        debug!("Got box {:?}", full_frame.box_info);

        if self.sinfo.positions {
            self.expect(IMDMessageType::FCoords, Some(self.n_atoms as i32))?;
            Self::read_into_array2(
                &mut self.reader,
                &mut full_frame.positions.as_mut().unwrap(),
                self.sinfo.endianness,
            )?;
        }
        if self.sinfo.velocities {
            self.expect(IMDMessageType::Velocities, Some(self.n_atoms as i32))?;
            Self::read_into_array2(
                &mut self.reader,
                &mut full_frame.velocities.as_mut().unwrap(),
                self.sinfo.endianness,
            )?;
        }
        if self.sinfo.forces {
            self.expect(IMDMessageType::Forces, Some(self.n_atoms as i32))?;
            Self::read_into_array2(
                &mut self.reader,
                &mut full_frame.forces.as_mut().unwrap(),
                self.sinfo.endianness,
            )?;
        }
        Ok(full_frame)
    }

    fn expect(
        &mut self,
        expected_type: IMDMessageType,
        expected_value: Option<i32>,
    ) -> io::Result<()> {
        self.header_buf.refill_from_reader(&mut self.reader)?;

        if self.header_buf.header_type != expected_type {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Expected header type {:?}, got {:?}",
                    expected_type, self.header_buf.header_type
                ),
            ));
        }

        if let Some(exp) = expected_value {
            if self.header_buf.length != exp {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Expected header length {}, got {}",
                        exp, self.header_buf.length
                    ),
                ));
            }
        }

        Ok(())
    }

    fn read_into_array2(
        reader: &mut impl Read,
        arr: &mut Array2<f32>,
        endian: Endianness,
    ) -> std::io::Result<()> {
        let slice = arr.as_slice_memory_order_mut().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Array not contiguous")
        })?;

        let byte_buf = unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr() as *mut u8,
                slice.len() * std::mem::size_of::<f32>(),
            )
        };

        reader.read_exact(byte_buf)?;

        if let Endianness::Big = endian {
            for val in slice.iter_mut() {
                *val = f32::from_be_bytes(val.to_ne_bytes());
            }
        }

        Ok(())
    }
}
#[derive(Debug)]
pub struct IMDClient {
    state: Arc<(Mutex<Shared>, Condvar)>,
    sinfo: IMDSessionInfo,
    producer_handle: Option<thread::JoinHandle<io::Result<()>>>,
    full_recv: Receiver<IMDFrame>,
    empty_send: Sender<IMDFrame>,
    error_recv: Receiver<io::Error>,
    prev_frame: Option<IMDFrame>,
    conn: TcpStream,
}

impl IMDClient {
    pub fn new<A: ToSocketAddrs>(
        addr: A,
        n_atoms: u64,
        buffer_size: Option<u64>,
    ) -> io::Result<Self> {
        let pause_proportion = 0.9;

        debug!("Connecting!");
        let mut conn = TcpStream::connect(&addr)?;

        debug!("Connected, waiting for handshake!");

        let sinfo = Self::await_imd_handshake(&mut conn)?;

        debug!("Handshake succeeded");

        let max_bytes = match buffer_size {
            Some(size) => size,
            None => 10 * 1024u64.pow(2),
        };
        let frame_bytes = sinfo.frame_size_bytes(n_atoms);
        let num_frames = max_bytes / frame_bytes;

        let (full_send, full_recv) = unbounded::<IMDFrame>();
        let (empty_send, empty_recv) = unbounded::<IMDFrame>();
        let (error_send, error_recv) = unbounded::<io::Error>();

        debug!("Created channels");

        for _ in 0..num_frames {
            empty_send
                .send(IMDFrame::new(n_atoms, sinfo))
                .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;
        }

        debug!("Loading channels");

        let producer_conn = conn.try_clone()?;

        debug!("Cloned socket");
        // debug!("Getting a frame {:?}", empty_recv.recv());

        let shared = Shared {
            consumer_finished: false,
        };
        let state = Arc::new((Mutex::new(shared), Condvar::new()));
        let producer_state = Arc::clone(&state);

        let producer_handle = Self::start_producer_thread(
            producer_state,
            producer_conn,
            empty_recv,
            full_send,
            num_frames,
            error_send,
            sinfo,
            n_atoms,
        );

        debug!("Started producer");

        IMDHeader::go().write_to(&mut conn)?;

        debug!("Sending go");

        // let duration = Duration::from_secs_f32(5.0);
        // thread::sleep(duration);
        // match producer_handle.join() {
        //     Ok(result) => debug!("Thread finished with result: {:?}", result),
        //     Err(e) => debug!("Thread panicked: {:?}", e),
        // }

        Ok(IMDClient {
            state,
            sinfo,
            producer_handle,
            full_recv,
            empty_send,
            error_recv,
            prev_frame: None,
            conn,
        })
    }

    fn notify_consumer_finished(&self) -> io::Result<()> {
        let (lock, cvar) = &*self.state;
        let mut shared = lock
            .lock()
            .map_err(|_poisoned| io::Error::new(ErrorKind::InvalidData, "lock poisoned"))?;
        shared.consumer_finished = true;
        cvar.notify_all();
        Ok(())
    }

    pub fn get_imdframe(&mut self) -> io::Result<&mut IMDFrame> {
        let new_imdframe = match self.full_recv.recv() {
            Ok(frame) => frame,

            Err(e) => {
                self.notify_consumer_finished()?;

                return Err(io::Error::new(ErrorKind::InvalidData, e));
            }
        };

        if let Some(prev) = self.prev_frame.take() {
            match self.empty_send.send(prev) {
                Err(e) => {
                    self.notify_consumer_finished()?;

                    return Err(io::Error::new(ErrorKind::InvalidData, e));
                }
                Ok(()) => {
                    let (_, cvar) = &*self.state;
                    cvar.notify_one();
                }
            }
        }

        self.prev_frame = Some(new_imdframe);

        self.prev_frame
            .as_mut()
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "Failed to retrieve IMDFrame"))
    }

    pub fn get_imdsessioninfo(&mut self) -> io::Result<IMDSessionInfo> {
        Ok(self.sinfo)
    }

    pub fn stop(&mut self) -> io::Result<()> {
        // best effort disconnect
        let _ = IMDHeader::disconnect().write_to(&mut self.conn);

        if let Some(handle) = self.producer_handle.take() {
            match handle.join() {
                Ok(inner) => inner,
                Err(e) => Err(io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Producer thread panicked: {:?}", e),
                )),
            }
        } else {
            Err(io::Error::new(
                std::io::ErrorKind::Other,
                "Producer thread already stopped",
            ))
        }
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

        debug!("Got header!");

        debug!("{:?}", header);

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
                // if we had to swap it for it to make sense, we're reading little
                // (since we read it as big)
                endianness = Some(Endianness::Little);
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
        state: Arc<(Mutex<Shared>, Condvar)>,
        conn: TcpStream,
        empty_recv: Receiver<IMDFrame>,
        full_send: Sender<IMDFrame>,
        num_frames: u64,
        error_send: Sender<io::Error>,
        sinfo: IMDSessionInfo,
        n_atoms: u64,
    ) -> Option<thread::JoinHandle<Result<(), io::Error>>> {
        Some(thread::spawn(move || -> Result<(), io::Error> {
            let mut producer = IMDProducer::new(
                state, conn, empty_recv, full_send, num_frames, error_send, sinfo, n_atoms,
            )?;
            producer.run()
        }))
    }
}

impl Drop for IMDClient {
    fn drop(&mut self) {
        debug!("RustIMDClient dropped (socket likely closing!)");
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
        listener.set_nonblocking(!blocking)?;
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
    use std::sync::Once;

    use super::IMDEnergies;
    use super::IMDTime;

    use super::IMDClient as RustIMDClient;
    use super::IMDFrame as RustIMDFrame;
    use super::IMDSessionInfo as RustIMDSessionInfo;
    use log::debug;
    use ndarray::ArrayView2;
    use numpy::PyArray2;
    use numpy::ToPyArray;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use pyo3::Bound;

    #[pyclass]
    #[derive(Debug)]
    pub struct IMDClient {
        inner: RustIMDClient,
    }
    #[pyclass(dict)]
    pub struct IMDFrame {
        #[pyo3(get)]
        step: Option<i64>,
        #[pyo3(get)]
        dt: Option<f64>,
        #[pyo3(get)]
        time: Option<f64>,
        #[pyo3(get)]
        energies: Option<Py<PyDict>>,
        #[pyo3(get)]
        positions: Option<Py<PyArray2<f32>>>,
        #[pyo3(get)]
        velocities: Option<Py<PyArray2<f32>>>,
        #[pyo3(get)]
        forces: Option<Py<PyArray2<f32>>>,
        #[pyo3(get)]
        r#box: Option<Py<PyArray2<f32>>>,
    }

    impl IMDFrame {
        pub fn from_rust<'py>(
            py: Python<'py>,
            rust_frame: &RustIMDFrame,
            anchor: Bound<'py, PyAny>,
        ) -> PyResult<Self> {
            let (step, dt, time) = match &rust_frame.time {
                Some(t) => (Some(t.step), Some(t.dt), Some(t.time)),
                None => (None, None, None),
            };

            let energies = rust_frame.energies.map(|e| {
                let dict = PyDict::new(py);
                dict.set_item("step", e.step).unwrap();
                dict.set_item("temperature", e.temperature).unwrap();
                dict.set_item("total", e.total).unwrap();
                dict.set_item("potential", e.potential).unwrap();
                dict.set_item("vdw", e.vdw).unwrap();
                dict.set_item("coulomb", e.coulomb).unwrap();
                dict.set_item("bonds", e.bonds).unwrap();
                dict.set_item("angles", e.angles).unwrap();
                dict.set_item("dihedrals", e.dihedrals).unwrap();
                dict.set_item("impropers", e.impropers).unwrap();
                dict.into()
            });

            let positions = match &rust_frame.positions {
                Some(p) => Some(unsafe { PyArray2::borrow_from_array(p, anchor.clone()).into() }),
                None => None,
            };

            let velocities = match &rust_frame.velocities {
                Some(v) => Some(unsafe { PyArray2::borrow_from_array(v, anchor.clone()).into() }),
                None => None,
            };

            let forces = match &rust_frame.forces {
                Some(f) => Some(unsafe { PyArray2::borrow_from_array(f, anchor.clone()).into() }),
                None => None,
            };

            let r#box = match &rust_frame.box_info {
                Some(f) => Some(unsafe { PyArray2::borrow_from_array(f, anchor.clone()).into() }),
                None => None,
            };

            Ok(Self {
                step,
                dt,
                time,
                energies,
                positions,
                velocities,
                forces,
                r#box,
            })
        }
    }
    #[pyclass(dict)]
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

    // pub fn make_imdframe_pyclass<'py>(
    //     py: Python<'py>,
    //     anchor: Bound<'py, PyAny>,
    //     rust_frame: &RustIMDFrame,
    // ) -> PyResult<Bound<'py, IMDFrame>> {
    //     let dict = pyo3::types::PyDict::new(py);

    //     if let Some(p) = &rust_frame.positions {
    //         let pyarray = unsafe { PyArray2::borrow_from_array(p, anchor.clone()) };
    //         dict.set_item("positions", pyarray)?;
    //     }

    //     if let Some(v) = &rust_frame.velocities {
    //         let pyarray = unsafe { PyArray2::borrow_from_array(v, anchor.clone()) };
    //         dict.set_item("velocities", pyarray)?;
    //     }

    //     if let Some(f) = &rust_frame.forces {
    //         let pyarray: Bound<'_, numpy::PyArray<f32, ndarray::Dim<[usize; 2]>>> =
    //             unsafe { PyArray2::borrow_from_array(f, anchor.clone()) };
    //         dict.set_item("forces", pyarray)?;
    //     }

    //     let frame = IMDFrame {
    //         dict: dict.unbind(),
    //     };

    //     Ok(Py::new(py, frame)?.into_bound(py))
    // }

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
                let client: RustIMDClient = RustIMDClient::new(addr, n_atoms, buffer_size)?;
                debug!("Client created: {:?}", client);

                Ok(IMDClient { inner: client })
            })
        }
        fn get_imdframe<'py>(
            this: Bound<'py, Self>,
            py: Python<'py>,
        ) -> PyResult<Bound<'py, IMDFrame>> {
            debug!("Borrowing client");
            let rust_client = &mut this.borrow_mut().inner;

            debug!("Got client, getting frame");

            let rust_frame = py
                .allow_threads(|| rust_client.get_imdframe())
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyEOFError, _>(format!("IO error: {}", e))
                })?;

            debug!("{:?}", rust_frame);

            let anchor = this.into_any();

            let imd_frame = IMDFrame::from_rust(py, &rust_frame, anchor)?;

            let py_frame = Py::new(py, imd_frame)?;

            Ok(py_frame.into_bound(py))
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

        fn stop<'py>(this: Bound<'py, Self>, py: Python<'py>) {
            let err = this.borrow_mut().inner.stop().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyEOFError, _>(format!("IO error: {}", e))
            });
            debug!("Shutdown reason: {:?}", err);
        }

        // pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        //     slf
        // }
    }

    #[pymodule]
    fn quickstream(m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
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
//             edebug!("Error reading from stream: {}", e);
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
//             edebug!("Error reading from stream: {}", e);
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

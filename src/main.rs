use byteorder::{BigEndian, ByteOrder, ReadBytesExt};
use log::{debug, error, info, trace, warn};
use std::{
    io::{self, ErrorKind, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    thread,
    time::Duration,
};

#[repr(i32)]
#[derive(Debug, PartialEq, Eq)]
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

fn get_header(conn: &mut TcpStream) -> io::Result<IMDHeader> {
    let header_type_val = conn.read_i32::<BigEndian>()?;
    let length = conn.read_i32::<BigEndian>()?;

    let header_type = IMDMessageType::try_from(header_type_val)
        .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;

    Ok(IMDHeader {
        header_type,
        length,
    })
}

fn get_imdsessioninfo(conn: &mut TcpStream, endianness: Endianness) -> io::Result<IMDSessionInfo> {
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

const IMDHEADERSIZE: usize = 8;
const IMDVERSIONS: &[i32] = &[2, 3];

#[derive(Debug)]
struct IMDHeader {
    header_type: IMDMessageType,
    length: i32,
}

// impl IMDHeader {
//     // fn from_bytes(buf: &[u8; IMDHEADERSIZE]) -> io::Result<Self> {
//     //     let header_type_val = i32::from_be_bytes(buf[0..4].try_into().unwrap());
//     //     let length = i32::from_be_bytes(buf[4..8].try_into().unwrap());

//     //     let header_type = IMDMessageType::try_from(header_type_val)
//     //         .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;

//     //     Ok(IMDHeader {
//     //         header_type,
//     //         length,
//     //     })
//     // }

//     fn new(header_type: i32, length: i32) {

//     }
// }

#[derive(Debug, Clone)]
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

fn main() {
    println!("Hello, world!");
}

pub struct IMDFrame;

pub struct IMDProducerV3;

pub struct IMDFrameBuffer;

pub enum IMDProducer {
    V3(IMDProducerV3),
}

pub struct IMDClient {
    sinfo: IMDSessionInfo,
    producer: IMDProducer,
}

impl IMDClient {
    pub fn new<A: ToSocketAddrs>(addr: A) -> io::Result<Self> {
        let mut conn = TcpStream::connect(&addr)?;

        let sinfo = IMDClient::await_imd_handshake(&mut conn)?;

        Ok(IMDClient {
            sinfo: sinfo,
            producer: IMDProducer::V3(IMDProducerV3),
        })
    }

    pub fn get_imdframe() -> io::Result<IMDFrame> {
        unimplemented!()
    }

    pub fn get_imdsessioninfo() -> io::Result<IMDSessionInfo> {
        unimplemented!()
    }

    pub fn stop() {
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

        let header = get_header(conn)?;

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
            let header = get_header(conn)?;

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

            return get_imdsessioninfo(conn, endianness);
        } else {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Unsupported IMD version: {}", version),
            ));
        };
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
    use super::IMDClient as RustIMDClient;
    use pyo3::prelude::*;

    #[pyclass]
    pub struct IMDClient {
        inner: RustIMDClient,
    }

    #[pymethods]
    impl IMDClient {
        #[new]
        fn new(host: &str, port: u16) -> PyResult<Self> {
            let addr = format!("{}:{}", host, port);
            let client = RustIMDClient::new(addr)?;
            Ok(IMDClient { inner: client })
        }

        fn get_imdframe(&self) -> PyResult<()> {
            unimplemented!()
        }

        fn get_imdsessioninfo(&self) -> PyResult<()> {
            unimplemented!()
        }
    }

    #[pymodule]
    fn quickstream(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<IMDClient>()?;
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

    #[test]
    fn server_client() {
        let mut imdserver = IMDServer::new("127.0.0.1:0", true).unwrap();
        let port = imdserver.port;

        let handle = thread::spawn(move || {
            let client_addr = imdserver.try_accept();
            // imdserver.
        });

        let address = format!("127.0.0.1:{}", port);
        let mut imdclient = IMDClient::new(address);
    }

    #[test]
    fn server_and_client_connect() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind listener");

        let addr = listener.local_addr().unwrap();
        let ip = addr.ip();
        let port = addr.port();

        // 2) Spawn the server thread that will accept one connection
        let handle = thread::spawn(move || {
            // blocks until a client connects
            let (mut server_stream, client_addr) =
                listener.accept().expect("failed to accept connection");

            assert_eq!(client_addr.ip().to_string(), "127.0.0.1");

            // Echo one message back to client
            let mut buf = [0u8; 5];
            server_stream.read_exact(&mut buf).unwrap();
            assert_eq!(&buf, b"hello");
            server_stream.write_all(b"world").unwrap();
        });

        let mut imdclient = IMDClient::new(addr);
        // assert_eq!(client_stream.peer_addr().unwrap(), addr);

        // // Send a small message and read the echo
        // client_stream.write_all(b"hello").unwrap();
        // let mut resp = [0u8; 5];
        // client_stream.read_exact(&mut resp).unwrap();
        // assert_eq!(&resp, b"world");

        // 4) Wait for the server thread to finish
        // handle.join().expect("server thread panicked");
    }
}

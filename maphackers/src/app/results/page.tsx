'use client';

import { Box, Button, Checkbox, Image, SimpleGrid, Stack, Switch } from "@mantine/core";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const [image, setImage] = useState('https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/images/bg-7.png');

  const [buildingOverlay, setBuildingOverlay] = useState('https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/images/bg-7.png');
  const [greenOverlay, setGreenOverlay] = useState('https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/images/bg-7.png');
  const [waterOverlay, setWaterOverlay] = useState('https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/images/bg-7.png');

  const [showBuilding, setShowBuilding] = useState(false);
  const [showGreen, setShowGreen] = useState(false);
  const [showWater, setShowWater] = useState(false);

  const [clickPosition, setClickPosition] = useState({ x: 0, y: 0 });
  const [zoomLevel, setZoomLevel] = useState(1);

  const imageRef = useRef<HTMLDivElement>(null);

  const handleClick = (e: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    if (zoomLevel === 2) {
      setZoomLevel(1);
      clickPosition.x = 0;
      clickPosition.y = 0;
      return;
    }

    const rect = (e.target as HTMLDivElement).getBoundingClientRect();
    const x = e.clientX - rect.left; // x position within the image
    const y = e.clientY - rect.top;  // y position within the image
    setClickPosition({ x, y });
    setZoomLevel(zoomLevel % 2 + 1);
  };

  useEffect(() => {
    fetch('//localhost:8000/results', {
      method: 'GET',
    })
      .then(response => response.json())
      .then(data => {
        setImage(data.image);
        setBuildingOverlay(data.building_overlay);
        setGreenOverlay(data.green_overlay);
        setWaterOverlay(data.lake_overlay);
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }, []);

  return (
    <>
      <div className="m-3">
        <Button m={3} component="a" href="/">Return to Upload</Button>
      </div>
      <div className="m-3">
        <SimpleGrid cols={2}>
          <div style={{ overflow: 'hidden', borderRadius: '500px' }}>
            <Box style={{
              position: 'relative',
              transform: `translate(-${imageRef.current ? (clickPosition.x / imageRef.current.offsetWidth) * 100 : 0}%, -${imageRef.current ? (clickPosition.y / imageRef.current.offsetHeight) * 100 : 0}%) scale(${zoomLevel})`,
            }} onClick={handleClick} ref={imageRef}>
              <Image
                src={image}
                alt="Background"
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                }}
              />
              <Image
                src={greenOverlay}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: 'auto',
                }}
                display={showGreen ? 'block' : 'none'}
              />
              <Image
                src={buildingOverlay}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: 'auto',
                }}
                display={showBuilding ? 'block' : 'none'}
              />
              <Image
                src={waterOverlay}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: 'auto',
                }}
                display={showWater ? 'block' : 'none'}
              />
            </Box>
          </div>
          <div>
            <Stack>
              <Switch
                label="Show Building Overlay"
                checked={showBuilding}
                onChange={() => setShowBuilding(!showBuilding)}
              />
              <Switch
                label="Show Green Overlay"
                checked={showGreen}
                onChange={() => setShowGreen(!showGreen)}
              />
              <Switch
                label="Show Water Overlay"
                checked={showWater}
                onChange={() => setShowWater(!showWater)}
              />
            </Stack>
          </div>
        </SimpleGrid>
      </div>
    </>
  );
}

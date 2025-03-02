'use client';

import { Button, Checkbox, Image, SimpleGrid, Stack } from "@mantine/core";
import { useEffect, useState } from "react";

export default function Home() {
  const [image, setImage] = useState('https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/images/bg-7.png');
  const [buildingOverlay, setBuildingOverlay] = useState(false);
  const [greenOverlay, setGreenOverlay] = useState(false);
  const [waterOverlay, setWaterOverlay] = useState(false);
  
  useEffect(() => {
    fetch('//localhost:8000/results', {
      method: 'GET',
    })
      .then(response => response.json())
      .then(data => {
        setImage(data.image);
        setBuildingOverlay(data.buildingOverlay);
        setGreenOverlay(data.greenOverlay);
        setWaterOverlay(data.waterOverlay);
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
          <div>
            <Image
              src={image}
            />
          </div>
          <div>
            <Stack>
              <Checkbox label="Building" />
              <Checkbox label="Green" />
              <Checkbox label="Water" />
            </Stack>
          </div>
        </SimpleGrid>
      </div>
    </>
  );
}

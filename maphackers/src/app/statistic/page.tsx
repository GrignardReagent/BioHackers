'use client';

import { Box, Button, Checkbox, FileInput, Image, SimpleGrid, Stack, Switch } from "@mantine/core";
import { useEffect, useRef, useState } from "react";
import { BarChart } from '@mantine/charts';

export default function Home() {
  const [image, setImage] = useState('https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/images/bg-7.png');
  const [classCount, setClassCount] = useState<{ key: string; value: unknown }[]>([]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const file = formData.get('image') as File;

    if (file) {
      try {
        const reader = new FileReader();
        reader.onloadend = () => {
          setImage(reader.result as string);
        };
        reader.readAsDataURL(file);

        const response = await fetch('http://localhost:8000/statistic', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('File upload failed');
        } else {
          const data = await response.json();
          console.log(data);
          setClassCount(Object.entries(data).map(([key, value]) => ({ key, value })));
          console.log(classCount);
        }
      } catch (error) {
        console.error('Error uploading file:', error);
        alert('Error uploading file');
      }
    } else {
      alert('Please select a file to upload');
    }
  };

  return (
    <>
      <form className="m-3" onSubmit={handleSubmit}>
        <FileInput
          m={3}
          placeholder="Upload image"
          name="image"
        />
        <Button m={3} type="submit">Upload</Button>
      </form>
      <div className="m-3">
        <SimpleGrid cols={2}>
          <div style={{ overflow: 'hidden', borderRadius: '500px' }}>
            <Box style={{
              position: 'relative',
            }}>
              <Image
                src={image}
                alt="Background"
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                }}
              />
            </Box>              
          </div>
          <div>
            <BarChart
              data={classCount}
              color="blue"
              style={{ height: 300 }} dataKey={"key"} series={[{name: 'value', color: 'violet.6'}]}/>
          </div>
        </SimpleGrid>
      </div>
    </>
  );
}
